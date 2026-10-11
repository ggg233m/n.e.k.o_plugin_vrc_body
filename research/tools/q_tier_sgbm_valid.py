# -*- coding: utf-8 -*-
r"""投票像素处的 **SGBM 有效性** —— 幻影机理的另一半。

本仓根因：低纹理/遮挡 ⇒ SGBM 给不出有效视差 ⇒ 视差随机 ⇒ δz=z²/(f·b)·δd 放大。
纹理那条量的是"像不像有结构"；这条量的是**SGBM 自己认不认这块**。

做法：对每格，取给它投障碍票的像素，回查该帧 SGBM 视差图在这些像素上有多少有效
（d>1）。若 B 类（无背书开洞）的投票像素大量落在 **SGBM 无效区**，
那它的票本身就是噪声产物 —— 这是与规则完全独立的判据。
"""
from __future__ import annotations
import argparse, dataclasses, sys
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research" / "tools"))
from backend.nav_grid import FREE, OCC, UNK  # noqa: E402
from backend.nav_mapping import make_sgbm, stereo_disparity  # noqa: E402
import mapping_gate as MG                       # noqa: E402

FX = 202.5
W, H = 720, 405
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0


def load_poses(rec):
    fp = rec / "final_poses.npz"
    if fp.exists():
        z = np.load(fp)
        return {int(i): np.asarray(T, np.float64) for i, T in zip(z["ids"], z["T"])}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--cells", type=int, default=120)
    a = ap.parse_args()
    rec = ROOT / "navmesh_recordings" / a.rec
    cfg_rec, info = MG.recorded_config(rec)
    got = {}
    for tag, qt in (("off", False), ("on", True)):
        ng, trail, st, m = MG.replay(rec, dataclasses.replace(cfg_rec, q_tiers=qt))
        walked = m.walked()
        got[tag] = (ng, ng.walked_mask(walked, 0.20), m)
    ng_off, wm, mapper = got["off"]
    ng_on = got["on"][0]
    if ng_off.grid.shape != ng_on.grid.shape:
        print("SHAPE MISMATCH"); return 2
    cam_h = float(mapper.cam_h)
    poses = load_poses(rec)
    kids = sorted(int(p.name.split("_")[0]) for p in (rec / "kf_img").glob("*_L.png"))
    print("rec %s  kf_img=%d" % (a.rec, len(kids)))
    matcher = make_sgbm(64)
    dcache = {}
    def disp_of(k):
        if k not in dcache:
            Lp = rec / "kf_img" / ("%06d_L.png" % k)
            Rp = rec / "kf_img" / ("%06d_R.png" % k)
            gL = cv2.imread(str(Lp), cv2.IMREAD_GRAYSCALE)
            gR = cv2.imread(str(Rp), cv2.IMREAD_GRAYSCALE)
            dcache[k] = None if (gL is None or gR is None) else stereo_disparity(gL, gR, make_sgbm(64))
        return dcache[k]

    g_off, g_on = ng_off.grid, ng_on.grid
    cleared = (g_off == OCC) & (g_on != OCC)
    kinds = {
        "B clear->FREE": cleared & (g_on == FREE) & ~wm,
        "C clear->UNK": cleared & (g_on == UNK) & ~wm,
        "retained": (g_off == OCC) & (g_on == OCC),
    }
    m2 = ng_on.meta; Hh, Wg = g_on.shape
    rng = np.random.default_rng(5)
    print()
    print("%-16s %6s %9s %12s" % ("class", "cells", "pixels", "SGBM invalid%"))
    res = {}
    for name, mask in kinds.items():
        idx = np.flatnonzero(mask.reshape(-1))
        if not len(idx):
            continue
        order = rng.permutation(idx)
        inv, tot, used = 0, 0, 0
        for i in order:
            if used >= a.cells:
                break
            r, c = int(i // Wg), int(i % Wg)
            x0 = m2.origin_xy_m[0] + c * m2.resolution_m
            y0 = m2.origin_xy_m[1] + (Hh - 1 - r) * m2.resolution_m
            best = None
            for k in kids[::9]:
                f = rec / "kf" / ("%06d.npz" % k)
                if not f.exists():
                    continue
                pts = np.load(f)["pts"].astype(np.float64)
                T = poses.get(k)
                if T is None or not len(pts):
                    continue
                hom = np.column_stack([pts, np.ones(len(pts))])
                pm = (T @ hom.T).T[:, :3]
                mm = (pm[:, 0] >= x0) & (pm[:, 0] < x0 + m2.resolution_m) & \
                     (pm[:, 1] >= y0) & (pm[:, 1] < y0 + m2.resolution_m)
                if not mm.any():
                    continue
                p = pts[mm]
                zb, xb, yb = p[:, 0], p[:, 1], p[:, 2]
                h = cam_h - (-yb)
                band = (zb > 0.3) & (h > 0.30) & (h < 2.0)
                if not band.any():
                    continue
                n = int(band.sum())
                if best is None or n > best[1]:
                    best = (k, n, zb[band], xb[band], yb[band])
            if best is None:
                continue
            k, n, zb, xb, yb = best
            d = disp_of(k)
            if d is None:
                continue
            u = CX - xb * FX / zb
            v = CY - yb * FX / zb
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not inb.any():
                continue
            ui = np.clip(np.round(u[inb]).astype(int), 0, W - 1)
            vi = np.clip(np.round(v[inb]).astype(int), 0, H - 1)
            dd = d[vi, ui]
            inv += int((dd <= 1.0).sum())
            tot += len(dd)
            used += 1
        if tot:
            res[name] = 100.0 * inv / tot
            print("%-16s %6d %9d %11.1f%%" % (name, used, tot, 100.0 * inv / tot))
    print()
    for k2 in ("B clear->FREE", "retained"):
        if k2 in res:
            print("  %-16s SGBM invalid = %.1f%%" % (k2, res[k2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
