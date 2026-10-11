# -*- coding: utf-8 -*-
r"""用双目原图测**投票像素处的纹理** —— 幻影机理的独立证据。

为什么这条与票型/距离分层正交：本仓的根因是
    * 低纹理平面 → SGBM 无匹配 → 视差随机；
    * δz = z²/(f·b)·δd ⇒ 同样的视差噪声在远处放大成米级高度误差。
两条都指向**图像本身**，而规则（`q_tiers`）只按**观测距离**给票分带，
从不看图像内容 ⇒ 纹理是规则之外的一维。

做法：对每个格，找出给它投障碍票的那些**像素**（精确反投影，约定已用
视差自洽校验过：(sx,sy)=(-1,-1) 时 z_implied/z 中位 1.000、99.9% 在 10% 内），
在该像素邻域测纹理（Sobel 梯度幅值中位 + 局部标准差），按类比。

预登记判据：若 B（无背书开洞）确由幻影构成 ⇒ 其纹理应**低于** retained（真结构）；
若两者不可区分 ⇒ 纹理这条对 q_tiers 无分辨力（如实报告）。
"""
from __future__ import annotations
import argparse, dataclasses, sys
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research" / "tools"))
from backend.nav_grid import FREE, OCC, UNK  # noqa: E402
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


def voting_pixels(rec, poses, k, x0, y0, res, cam_h):
    f = rec / "kf" / ("%06d.npz" % k)
    if not f.exists():
        return None
    zz = np.load(f)
    pts = zz["pts"].astype(np.float64)
    T = poses.get(k)
    if T is None or len(pts) == 0:
        return None
    hom = np.column_stack([pts, np.ones(len(pts))])
    pm = (T @ hom.T).T[:, :3]
    m = (pm[:, 0] >= x0) & (pm[:, 0] < x0 + res) & (pm[:, 1] >= y0) & (pm[:, 1] < y0 + res)
    if not m.any():
        return None
    p = pts[m]
    zb, xb, yb = p[:, 0], p[:, 1], p[:, 2]
    h = cam_h - (-yb)
    band = (zb > 0.3) & (h > 0.30) & (h < 2.0)
    if not band.any():
        return None
    zb, xb, yb = zb[band], xb[band], yb[band]
    u = CX - xb * FX / zb
    v = CY - yb * FX / zb
    inb = (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
    if not inb.any():
        return None
    return u[inb], v[inb]


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
    if not kids:
        print("no kf_img"); return 2
    print("rec %s  kf_img=%d  cam_h=%.3f" % (a.rec, len(kids), cam_h))

    # 预载灰度图 + Sobel 幅值（缓存）
    cache = {}
    def gray_mag(k):
        if k not in cache:
            p = rec / "kf_img" / ("%06d_L.png" % k)
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None:
                cache[k] = None
            else:
                gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
                cache[k] = (g.astype(np.float32), np.hypot(gx, gy))
        return cache[k]

    g_off, g_on = ng_off.grid, ng_on.grid
    cleared = (g_off == OCC) & (g_on != OCC)
    kinds = {
        "B clear->FREE": cleared & (g_on == FREE) & ~wm,
        "C clear->UNK": cleared & (g_on == UNK) & ~wm,
        "retained": (g_off == OCC) & (g_on == OCC),
    }
    m2 = ng_on.meta
    Hh, Wg = g_on.shape
    rng = np.random.default_rng(3)

    print()
    print("%-16s %6s %8s %10s %10s %10s" % ("class", "cells", "pixels", "grad_p50", "grad_p90", "frac<8"))
    results = {}
    for name, mask in kinds.items():
        idx = np.flatnonzero(mask.reshape(-1))
        if not len(idx):
            continue
        order = rng.permutation(idx)
        grads = []
        npix = 0
        used = 0
        for i in order:
            if used >= a.cells:
                break
            r, c = int(i // Wg), int(i % Wg)
            x0 = m2.origin_xy_m[0] + c * m2.resolution_m
            y0 = m2.origin_xy_m[1] + (Hh - 1 - r) * m2.resolution_m
            best = None
            for k in kids[::9]:
                vp = voting_pixels(rec, poses, k, x0, y0, m2.resolution_m, cam_h)
                if vp is None:
                    continue
                if best is None or len(vp[0]) > best[1]:
                    best = (k, len(vp[0]), vp)
            if best is None:
                continue
            k, n, (us, vs) = best
            gm = gray_mag(k)
            if gm is None:
                continue
            _, mag = gm
            ui = np.clip(np.round(us).astype(int), 0, W - 1)
            vi = np.clip(np.round(vs).astype(int), 0, H - 1)
            g = mag[vi, ui]
            grads.append(g)
            npix += len(g)
            used += 1
        if not grads:
            print("%-16s  none" % name); continue
        allg = np.concatenate(grads)
        results[name] = allg
        print("%-16s %6d %8d %10.2f %10.2f %9.1f%%"
              % (name, used, npix, float(np.median(allg)), float(np.percentile(allg, 90)),
                 100.0 * float(np.mean(allg < 8.0))))

    # 两两对照（Mann-Whitney 用中位差近似，不引 scipy.stats 也行）
    print()
    if "B clear->FREE" in results and "retained" in results:
        b, rr = results["B clear->FREE"], results["retained"]
        print("B vs retained: 中位 %.2f vs %.2f  (比值 %.3f)"
              % (np.median(b), np.median(rr), np.median(b) / max(np.median(rr), 1e-9)))
        try:
            from scipy.stats import mannwhitneyu
            u, p = mannwhitneyu(b, rr, alternative="less")
            print("  Mann-Whitney (B < retained): p = %.3g" % p)
        except Exception as exc:
            print("  (mannwhitney 不可用: %s)" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
