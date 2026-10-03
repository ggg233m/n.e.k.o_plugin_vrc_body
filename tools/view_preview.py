# -*- coding: utf-8 -*-
"""把"多层/净空视图"到底长什么样画出来（用现有数据，不假设任何新累加器）。

    python -m tools.view_preview [录制名]

两个候选，都是**俯视**、一格一像素，和 ``/navmesh`` 现在的画法一致：

  A 净空高度图   每格 = 正上方最近那张面的高度（surface_grid 的 mean_h），
                 连续色标。障碍仍是黑色。**需要的额外数据：零。**
  B 带切片       4 张并排，每张一条高度带（below / 2~3 / 3~4.5 / >4.5），
                 有票=亮、无票=空。**需要的额外数据：零**（band_grid 已够）。

真正要"多层"（一个格子里有几层）需要带内高度直方图，**尚未做**，这里画不出来。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_grid import OCC, UNK                  # noqa: E402
from backend.nav_mapping import KeyframeGridMapper, MapperConfig  # noqa: E402
from tools.precision_probe import load                 # noqa: E402

REC = "20261001_044153"
BAND_NAMES = ("below", "lo 2-3m", "mid 3-4.5m", "hi >4.5m")


def turbo_like(t: np.ndarray) -> np.ndarray:
    """0..1 → 冷(蓝)→暖(红) 色标；NaN 走灰。"""
    t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
    r = np.clip(1.6 * t - 0.35, 0, 1)
    g = np.clip(1.35 * np.sin(np.pi * np.clip(t, 0, 1)) ** 0.85, 0, 1)
    b = np.clip(1.25 - 2.1 * t, 0, 1)
    return (np.stack([b, g, r], -1) * 255).astype(np.uint8)


def fit(img: np.ndarray, width: int = 430) -> np.ndarray:
    h, w = img.shape[:2]
    sc = width / max(w, 1)
    if sc < 1.0:
        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_NEAREST)
    return img


def label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main() -> None:
    rec_name = sys.argv[1] if len(sys.argv) > 1 else REC
    frames = load(ROOT / "navmesh_recordings" / rec_name)
    if not frames:
        print("无关键帧")
        return
    m = KeyframeGridMapper(MapperConfig())
    for i, fr in enumerate(frames):
        m.add_keyframe(i, fr[1], fr[2])
    ng = m.rasterize()
    sg = m.surface_grid()
    bg = m.band_grid()
    g = ng.grid
    occ = g == OCC
    seen = g != UNK

    # ---- A 净空高度图 ----
    hgt = sg["mean_h"]
    v = np.isfinite(hgt)
    lo, hi = (float(np.nanpercentile(hgt, 2)), float(np.nanpercentile(hgt, 98))) if v.any() else (0, 1)
    t = (hgt - lo) / max(hi - lo, 1e-6)
    imgA = np.full((*hgt.shape, 3), 28, np.uint8)                 # 没看到头顶 = 深色（"没数据"）
    imgA[v] = turbo_like(t[v])
    imgA[seen & ~v] = (60, 60, 60)                              # 看到地面但头顶无票
    imgA[occ] = (20, 20, 20)                                    # 地面层障碍压黑
    A = label(fit(imgA), f"A  clearance map   {lo:.2f}-{hi:.2f} m   black = ground obstacle")

    # ---- B 带切片 ----
    tiles = []
    for i, name in enumerate(BAND_NAMES):
        a = bg[i]
        p = np.clip(np.log1p(a) / np.log1p(max(float(a.max()), 1.0)), 0, 1)
        im = np.full((*a.shape, 3), 24, np.uint8)
        im[a > 0.5] = (turbo_like(p * 0.75 + 0.25)[a > 0.5])
        im[occ] = (20, 20, 20)
        tiles.append(label(fit(im), f"B{i}  {name}"))
    gap = np.full((tiles[0].shape[0], 8, 3), 40, np.uint8)
    B = tiles[0]
    for t_ in tiles[1:]:
        B = np.hstack([B, gap, t_])

    W = max(A.shape[1], B.shape[1])
    if A.shape[1] < W:
        A = np.hstack([A, np.full((A.shape[0], W - A.shape[1], 3), 40, np.uint8)])
    if B.shape[1] < W:
        B = np.hstack([B, np.full((B.shape[0], W - B.shape[1], 3), 40, np.uint8)])
    canvas = np.vstack([A, np.full((10, W, 3), 40, np.uint8), B])
    out = ROOT / "tools" / f"view_{rec_name}.png"
    cv2.imwrite(str(out), canvas)
    print(f"录制 {rec_name}  {len(frames)} 关键帧  栅格 {g.shape}")
    print(f"  净空高度范围 {lo:.2f}–{hi:.2f} m，有头顶票的格 {int(v.sum())}")
    for i, name in enumerate(BAND_NAMES):
        a = bg[i]
        print(f"  带 {name:11s} 非空格 {int((a > 0.5).sum()):7d}  点数 {float(a.sum()):11.0f}")
    print(f"  地面层障碍 {int(occ.sum())}")
    print(f"  预览图：{out}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
