# -*- coding: utf-8 -*-
"""高度分带占用：真实录制上量覆盖，并出俯视图。

    python -m tools.band_report [录制名 ...]

回答"分层之后到底捡回了多少头顶之上的几何"：每条带有多少格见过东西、总共多少点，
以及**哪一条带在这个格是主导带**（俯视图上按主导带上色）。

    python -m tools.band_report 20261001_044153
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_grid import OCC, UNK                      # noqa: E402
from backend.nav_mapping import KeyframeGridMapper, MapperConfig  # noqa: E402
from tools.precision_probe import load                     # noqa: E402

RECS = ("20261001_044153", "20260929_045615")
BAND_COLOR = {           # BGR
    "below": (150, 100, 60),      # 地面以下（棕）
    "lo": (60, 170, 240),         # 头顶 2~3.5 m（橙）
    "mid": (200, 120, 60),        # 3.5~6 m（蓝）
    "hi": (60, 60, 220),          # >6 m（红）
    "ground": (190, 190, 190),    # 只有地面观测，没有高带票
}


def run(rec: Path) -> tuple[np.ndarray, np.ndarray, dict, list]:
    frames = load(rec)
    m = KeyframeGridMapper(MapperConfig(hi_bands=True))
    for i, fr in enumerate(frames):
        m.add_keyframe(i, fr[1], fr[2])
    ng = m.rasterize()
    bg = m.band_grid()
    bc = m.band_counts()
    stats = {"frames": len(frames), "res_m": bc["res_m"], "edges_m": bc["edges_m"],
             "bands": {}, "grid_cells": int(ng.grid.size),
             "ground_occ_cells": int((ng.grid == OCC).sum())}
    for i, name in enumerate(("below", "lo", "mid", "hi")):
        a = bg[i]
        stats["bands"][name] = {"cells": int((a > 0.5).sum()), "points": int(a.sum())}
    stats["bands_hi_total_cells"] = int(sum(v["cells"] for v in stats["bands"].values()))
    return ng.grid, bg, stats, [fr[2][:2, 3].copy() for fr in frames]


def render(ng: np.ndarray, bg: np.ndarray, trail, dest: Path) -> None:
    """主导带上色：每格取点数最多的那条带；地面层障碍用深色压一层，免得看起来全是空的。"""
    h, w = ng.shape
    img = np.full((h, w, 3), 245, np.uint8)
    counts = np.stack([bg[i] for i in range(4)])
    has_hi = counts.max(axis=0) > 0.5
    dom = counts.argmax(axis=0)
    for i, name in enumerate(("below", "lo", "mid", "hi")):
        img[has_hi & (dom == i)] = BAND_COLOR[name]
    seen = ng != UNK
    img[seen & ~has_hi] = BAND_COLOR["ground"]
    img[ng == OCC] = (30, 30, 30)                       # 地面层障碍（真墙）
    sc = min(1.0, 820 / max(h, w))
    if sc < 1.0:
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    bar = np.full((24, img.shape[1], 3), 45, np.uint8)
    cv2.putText(bar, "below  lo(2-3.5m)  mid(3.5-6m)  hi(>6m)  = ground-only   black = ground obstacle",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = np.vstack([bar, img])
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), canvas)


def main() -> None:
    names = sys.argv[1:] or list(RECS)
    for name in names:
        rec = ROOT / "navmesh_recordings" / name
        if not rec.exists():
            print(f"[skip] {name}")
            continue
        ng, bg, st, trail = run(rec)
        print(f"\n=== {name} ===  关键帧 {st['frames']}  栅格 {st['grid_cells']} 格 "
              f"(res {st['res_m']} m)  带边界(上界) {st['edges_m']}")
        tot = 0
        for k, v in st["bands"].items():
            pct = 100.0 * v["cells"] / max(st["grid_cells"], 1)
            print(f"  {k:6s} 非空格 {v['cells']:7d} ({pct:5.1f}%)   点数 {v['points']:10d}")
            tot += v["cells"]
        print(f"  高带非空格合计 {tot}（占全图 {100.0 * tot / max(st['grid_cells'], 1):.1f}%）")
        print(f"  地面层障碍格 {st['ground_occ_cells']}（未变，导航照旧）")
        render(ng, bg, trail, ROOT / "tools" / f"band_{name}.png")
        print(f"  俯视图：tools/band_{name}.png")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
