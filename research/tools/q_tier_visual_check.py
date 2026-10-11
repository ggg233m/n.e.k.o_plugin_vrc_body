# -*- coding: utf-8 -*-
r"""把被清格**反投影回双目原图**，做成对照表 —— 用眼睛看那一格到底是什么。

前裁定说"没有外部真值"。但 `kf_img/<id>_{L,R}.png` 是**与建图同一时刻的双目原图**，
而 `kf/<id>.npz` 的 `pts` 是同一帧的 SGBM 点云（base 系，追踪米）。两者可互相标定：

    p_base = inv(T_map) @ p_map
    u = cx - Y*fx/X        (FACTS §:  pts = [z,-x,-y] 光学系 ⇒ 等价 base 系)
    v = cy - Z*fy/X

⇒ 可以把任意地图格投到任意关键帧的画面上。于是"这格是不是真墙"变成**可看的问题**：
真墙在图上是有纹理/边缘的连续面；量化倾斜的幻影多半落在低纹理、无结构处。

本工具输出**对照拼图**：类 B（无背书开洞）/ 类 C（惰性）/ 保留（未被清）各若干格，
同一录制、同一套投影，肉眼可直接比。**不做自动判定** —— 那是人看的活。
"""
from __future__ import annotations
import argparse, dataclasses, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research" / "tools"))
from backend.nav_grid import FREE, OCC, UNK  # noqa: E402
import mapping_gate as MG                       # noqa: E402

CROP = 150
FX = 202.5


def cell_centers_map(ng):
    """每格中心的地图系坐标（追踪米），形状 (H, W, 2)。"""
    m = ng.meta; H, W = ng.grid.shape
    rr, cc = np.mgrid[0:H, 0:W]
    x = m.origin_xy_m[0] + (cc + 0.5) * m.resolution_m
    y = m.origin_xy_m[1] + (H - 1 - rr + 0.5) * m.resolution_m
    return np.stack([x, y], -1)


def project_to_kf(xy_map, T_map, W, H, cam_h=1.755, z_default=1.2):
    """地图系 (x,y) 投到该关键帧像素。返回 (u, v, ok)。

    格只有平面坐标、没有高度 ⇒ 取相机高度处（地面上方 cam_h）作为代表高度，
    即"这一格在相机视线里、与相机等高"的那个点。这是**展示用近似**，不是几何解。
    """
    Tinv = np.linalg.inv(T_map)
    p = np.array([xy_map[0], xy_map[1], 0.0, 1.0])      # 地图系，z=0（地面高度处）
    b = Tinv @ p                                         # base 系
    X, Y, Z = b[0], b[1], b[2] + cam_h                   # 抬高到相机高度
    if X <= 0.3:
        return -1.0, -1.0, False
    u = (W - 1) / 2.0 - Y * FX / X
    v = (H - 1) / 2.0 - Z * FX / X
    ok = (0 <= u < W) and (0 <= v < H)
    return u, v, ok


def pick_cells(ng_off, ng_on, wm, kind, rng, limit):
    g_off, g_on = ng_off.grid, ng_on.grid
    cleared = (g_off == OCC) & (g_on != OCC)
    if kind == "B":
        m = cleared & (g_on == FREE) & ~wm
    elif kind == "C":
        m = cleared & (g_on == UNK) & ~wm
    elif kind == "retained":
        m = (g_off == OCC) & (g_on == OCC)
    else:
        raise ValueError(kind)
    idx = np.flatnonzero(m.reshape(-1))
    if len(idx) == 0:
        return []
    sel = rng.choice(idx, size=min(limit, len(idx)), replace=False)
    H, W = m.shape
    return [(int(i // W), int(i % W)) for i in sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--out", default=".tmp/_clearcrops")
    a = ap.parse_args()
    from PIL import Image, ImageDraw

    rec = ROOT / "navmesh_recordings" / a.rec
    cfg_rec, info = MG.recorded_config(rec)
    got = {}
    for tag, qt in (("off", False), ("on", True)):
        ng, trail, st, m = MG.replay(rec, dataclasses.replace(cfg_rec, q_tiers=qt))
        walked = m.walked()
        wm = ng.walked_mask(walked, 0.20)
        got[tag] = (ng, wm, m)
    ng_off, wm, mapper = got["off"]
    ng_on = got["on"][0]
    if ng_off.grid.shape != ng_on.grid.shape:
        print("SHAPE MISMATCH"); return 2

    imgs = sorted((rec / "kf_img").glob("*_L.png"))
    if not imgs:
        print("没有 kf_img（该录制 record_images=False）"); return 2
    print("录制 %s  kf_img=%d 帧" % (a.rec, len(imgs)))

    # 关键帧位姿（优先 final_poses，与建图一致）
    fp = rec / "final_poses.npz"
    if fp.exists():
        z = np.load(fp)
        poses = {int(i): np.asarray(T, np.float64) for i, T in zip(z["ids"], z["T"])}
    else:
        poses = {}
    kids = sorted(int(p.name.split("_")[0]) for p in imgs)

    centers = cell_centers_map(ng_on)
    rng = np.random.default_rng(7)
    outdir = ROOT / a.out
    outdir.mkdir(parents=True, exist_ok=True)

    rows = {}
    for kind in ("B", "C", "retained"):
        cells = pick_cells(ng_off, ng_on, wm, kind, rng, a.per_class)
        tiles = []
        for (r, c) in cells:
            xy = centers[r, c]
            # 找最"正对"这一格的、有图的 kf
            best = None
            for k in kids:
                T = poses.get(k)
                if T is None:
                    kf = rec / "kf" / ("%06d.npz" % k)
                    if not kf.exists():
                        continue
                    T = np.load(kf)["T_map"].astype(np.float64)
                u, v, ok = project_to_kf(xy, T, 720, 405)
                if not ok:
                    continue
                # 偏好：离画面中心近 + 离相机近（X 小）
                Tinv = np.linalg.inv(T)
                X = (Tinv @ np.array([xy[0], xy[1], 0.0, 1.0]))[0]
                if X <= 0.4 or X > 6.0:
                    continue
                cost = abs(u - 360) / 360.0 + abs(v - 202) / 202.0 + X / 6.0
                if best is None or cost < best[0]:
                    best = (cost, k, u, v, X)
            if best is None:
                continue
            _, k, u, v, X = best
            src = rec / "kf_img" / ("%06d_L.png" % k)
            if not src.exists():
                continue
            im = Image.open(src).convert("RGB")
            u0 = int(np.clip(u - CROP // 2, 0, im.width - CROP))
            v0 = int(np.clip(v - CROP // 2, 0, im.height - CROP))
            tile = im.crop((u0, v0, u0 + CROP, v0 + CROP))
            d = ImageDraw.Draw(tile)
            cu, cv = u - u0, v - v0
            d.ellipse([cu - 9, cv - 9, cu + 9, cv + 9], outline=(255, 0, 0), width=2)
            d.text((3, 3), "kf%d %.1fm" % (k, X), fill=(255, 255, 0))
            tiles.append(tile)
        rows[kind] = tiles
        print("  %-9s 采到 %d 格" % (kind, len(tiles)))

    # 拼图：三行 × per-class 列
    ncol = max((len(v) for v in rows.values()), default=0)
    if ncol == 0:
        print("没有采到任何格"); return 2
    label_h = 22
    sheet = Image.new("RGB", (ncol * CROP, 3 * (CROP + label_h)), (20, 20, 24))
    dr = ImageDraw.Draw(sheet)
    for ri, kind in enumerate(("B", "C", "retained")):
        y0 = ri * (CROP + label_h)
        dr.text((4, y0 + 4), "%s (n=%d)" % (kind, len(rows[kind])), fill=(255, 255, 255))
        for ci, t in enumerate(rows[kind]):
            sheet.paste(t, (ci * CROP, y0 + label_h))
    outp = outdir / ("sheet_%s.png" % a.rec)
    sheet.save(outp)
    print("写出 %s" % outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
