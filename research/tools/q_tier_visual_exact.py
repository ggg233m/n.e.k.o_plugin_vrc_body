# -*- coding: utf-8 -*-
r"""用**投票的那批点**做目视对照 —— 精确反投影版。

前一版（q_tier_visual_check.py）把格中心在"地面高度处"投到画面，是**近似**：
真正投票的障碍点在 0.3–2.0 m 的高度带里，不是地面。本版改成**精确投影**：

    对每个关键帧 k，把它的 pts（base 系，追踪米）用 T_map 投到地图系，
    落在目标格内的那些点**就是投票点**；它们本来就在 k 的 base 系里，
    所以直接按同一套内参投回 k 的画面即可 —— 无近似。

于是红圈标的是"这一格的障碍票是画面里的哪块像素投出来的"。
人眼对**纹理贫乏 / 重复纹理**区域仍然有效（那正是 SGBM 的失效区），
所以这是与票型、与高度矩都独立的第三条证据 —— 且是**语义**级别的。

⚠️ 本工具**不自动判定**。它只把证据摆出来给人看：
   真墙在图上是有纹理/边缘的连续面；量化倾斜的幻影多半落在无结构处。
"""
from __future__ import annotations
import argparse, dataclasses, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research" / "tools"))
from backend.nav_grid import FREE, OCC, UNK  # noqa: E402
import mapping_gate as MG                       # noqa: E402

FX = 202.5


def load_poses(rec):
    fp = rec / "final_poses.npz"
    if fp.exists():
        z = np.load(fp)
        return {int(i): np.asarray(T, np.float64) for i, T in zip(z["ids"], z["T"])}
    return {}


def pose_of(rec, poses, k):
    if k in poses:
        return poses[k]
    f = rec / "kf" / ("%06d.npz" % k)
    if f.exists():
        return np.load(f)["T_map"].astype(np.float64)
    return None


def voting_pixels(rec, poses, k, xy_cell, res, cam_h):
    """返回该 kf 里落在目标格内的障碍带点，投到画面的 (u,v) 列表 + 世界高度。"""
    f = rec / "kf" / ("%06d.npz" % k)
    if not f.exists():
        return None
    z = np.load(f)
    pts = z["pts"].astype(np.float64)
    T = pose_of(rec, poses, k)
    if T is None or len(pts) == 0:
        return None
    # base -> map（只关心 xy 落格）
    hom = np.column_stack([pts, np.ones(len(pts))])
    pm = (T @ hom.T).T[:, :3]
    x0, y0 = xy_cell
    m = (pm[:, 0] >= x0) & (pm[:, 0] < x0 + res) & (pm[:, 1] >= y0) & (pm[:, 1] < y0 + res)
    if not m.any():
        return None
    p = pts[m]                       # base 系：光学 z 前 / x 右 / y 下
    zz, xx, yy = p[:, 0], p[:, 1], p[:, 2]
    ok = zz > 0.3
    if not ok.any():
        return None
    zz, xx, yy = zz[ok], xx[ok], yy[ok]
    # 高度（离地）= cam_h - y_base   （y_base 向下为正）
    h = cam_h - (-yy)                # pts=[z,-x,-y] ⇒ y_base = -pts[:,2]
    band = (h > 0.30) & (h < 2.0)
    if not band.any():
        return None
    zz, xx, yy, h = zz[band], xx[band], yy[band], h[band]
    W, H = 720, 405
    u = (W - 1) / 2.0 - (-xx) * FX / zz      # 光学 x 右 ⇒ u = cx + x*f/z，而 pts[1]=-x
    v = (H - 1) / 2.0 - (-yy) * FX / zz
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inb.any():
        return None
    return dict(u=u[inb], v=v[inb], h=h[inb], z=zz[inb], n=int(inb.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--crop", type=int, default=190)
    a = ap.parse_args()
    from PIL import Image, ImageDraw

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
    print("录制 %s  cam_h=%.3f 追踪米  world_scale=%.3f" % (a.rec, cam_h, ng_on.meta.world_scale))

    kids = sorted(int(p.name.split("_")[0]) for p in (rec / "kf_img").glob("*_L.png"))
    if not kids:
        print("没有 kf_img"); return 2
    poses = load_poses(rec)

    g_off, g_on = ng_off.grid, ng_on.grid
    cleared = (g_off == OCC) & (g_on != OCC)
    kinds = {
        "B_clear->FREE": cleared & (g_on == FREE) & ~wm,
        "C_clear->UNK": cleared & (g_on == UNK) & ~wm,
        "retained_OCC": (g_off == OCC) & (g_on == OCC),
    }
    m2 = ng_on.meta
    H, Wg = g_on.shape
    rng = np.random.default_rng(11)

    rows = {}
    for name, mask in kinds.items():
        idx = np.flatnonzero(mask.reshape(-1))
        if not len(idx):
            rows[name] = []; continue
        order = rng.permutation(idx)
        tiles = []
        for i in order:
            if len(tiles) >= a.per_class:
                break
            r, c = int(i // Wg), int(i % Wg)
            x0 = m2.origin_xy_m[0] + c * m2.resolution_m
            y0 = m2.origin_xy_m[1] + (H - 1 - r) * m2.resolution_m
            # 找票最多的那个 kf
            best = None
            for k in kids[::7]:                       # 抽样加速；够用
                v = voting_pixels(rec, poses, k, (x0, y0), m2.resolution_m, cam_h)
                if v is None:
                    continue
                score = v["n"] - 0.02 * float(np.median(v["z"]))
                if best is None or score > best[0]:
                    best = (score, k, v)
            if best is None:
                continue
            _, k, v = best
            src = rec / "kf_img" / ("%06d_L.png" % k)
            if not src.exists():
                continue
            im = Image.open(src).convert("RGB")
            cu, cv = float(np.median(v["u"])), float(np.median(v["v"]))
            C = a.crop
            u0 = int(np.clip(cu - C // 2, 0, im.width - C))
            v0 = int(np.clip(cv - C // 2, 0, im.height - C))
            tile = im.crop((u0, v0, u0 + C, v0 + C))
            d = ImageDraw.Draw(tile)
            # 画出**全部投票点**（细点，看清它们散在多大范围）
            for uu, vv in zip(v["u"], v["v"]):
                d.point((uu - u0, vv - v0), fill=(0, 200, 255))
            du, dv = cu - u0, cv - v0
            # ⚠️ 关键：按**真实角尺寸**画格，不是固定半径。
            # 0.1 m 格在距离 z 处张 fx*0.1/z 像素（5 m 时只有 ~4 px）。
            # 画固定 10 px 会让人以为格很大、误判"圈住了什么结构"。
            zm = float(np.median(v["z"]))
            half = max(2.0, 0.5 * FX * 0.1 / max(zm, 1e-6))
            d.rectangle([du - half, dv - half, du + half, dv + half], outline=(255, 40, 40), width=1)
            # 投票点外包框（说明这张票散布多广 —— 散布大 = 不是一块面）
            d.rectangle([v["u"].min() - u0, v["v"].min() - v0,
                         v["u"].max() - u0, v["v"].max() - v0], outline=(255, 170, 0), width=1)
            d.text((3, 3), "kf%d n=%d h=%.2f" % (k, v["n"], float(np.median(v["h"]))), fill=(255, 255, 0))
            tiles.append(tile)
        rows[name] = tiles
        print("  %-16s 采到 %d 格" % (name, len(tiles)))

    ncol = max((len(t) for t in rows.values()), default=0)
    if not ncol:
        print("没采到"); return 2
    C, lh = a.crop, 22
    names = list(rows.keys())
    sheet = Image.new("RGB", (ncol * C, len(names) * (C + lh)), (18, 18, 22))
    dr = ImageDraw.Draw(sheet)
    for ri, name in enumerate(names):
        y0 = ri * (C + lh)
        dr.text((4, y0 + 5), name, fill=(255, 255, 255))
        for ci, t in enumerate(rows[name]):
            sheet.paste(t, (ci * C, y0 + lh))
    outdir = ROOT / ".tmp" / "_clearcrops"
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / ("exact_%s.png" % a.rec)
    sheet.save(outp)
    print("写出 %s" % outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
