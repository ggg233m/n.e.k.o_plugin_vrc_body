# -*- coding: utf-8 -*-
"""ray_clear 到底在去虚假障碍，还是在清真结构？——2×2 消融 + 走过证据。

    python -m tools.ray_clear_ab [录制名...]

## 要回答的

`low_ceiling_nav` 量到：沙发后面的矮墙（0.65~0.85 m、1.35 m 两道）被判成可走，
机制是 ``ray_clear`` 的看穿判据把它们的障碍票清零，逐层打中:看穿 = 2:55。
那么**关掉 ray_clear 会放回来多少**，以及**放回来的里面有多少是用户真走过的**。

## 走过的证据够不够硬

只用**中心线**（``line``，thickness=1，即头实际经过的那条线），不用加粗的
``walked_mask``（0.20 m 半宽，只代表"离得近"）。中心线穿过 OCC ⇒ 用户当时是从那儿
走过去的 ⇒ 那儿没有实心碰撞体。

**这条证据有已知的偏置**：人会本能绕开看得见的家具，所以"没走过"不代表不能过；
反过来隐形碰撞体用户毫无察觉，"走过"也不能 100% 证明能过。所以
**走过-OCC 只能算"可疑误判"的下界**，不是确证。表里给的是这个下界。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import KeyframeGridMapper, MapperConfig   # noqa: E402
from backend.nav_grid import FREE, OCC, UNK                        # noqa: E402
from tools.precision_probe import load                            # noqa: E402

RECS = ("20261001_044153", "20260929_045615", "20261001_044120")
COMBOS = ((True, True), (False, True), (True, False), (False, False))


def walked_lines(ng, walked) -> np.ndarray:
    """``NavGrid.build`` 里那两行 polylines 的原样复刻，只取中心线 ``line``。"""
    line = np.zeros(ng.grid.shape, np.uint8)
    for poly in walked:
        pts = np.array([ng.to_cell(p)[::-1] for p in np.asarray(poly, float)], np.int32)
        if len(pts) >= 2:
            cv2.polylines(line, [pts.reshape(-1, 1, 2)], False, 1, 1)
    return line


def one(frames, c: MapperConfig):
    m = KeyframeGridMapper(c)
    for i, fr in enumerate(frames):
        m.add_keyframe(i, fr[1], fr[2])
    ng = m.rasterize()
    ng.build(radius_m=0.25, walked=m.walked())
    g = ng.grid
    line = walked_lines(ng, m.walked())
    occ = g == OCC
    seen = (g == FREE) | occ
    return {"occ": int(occ.sum()), "free": int((g == FREE).sum()),
            "unk": int((g == UNK).sum()),
            "occ_walked": int((occ & (line > 0)).sum()),
            "line": int((line > 0).sum()),
            "walked_on_free": int(((g == FREE) & (line > 0)).sum()),
            "seen": int(seen.sum()),
            "ray_cleared": int(m.ray_cleared_cells),
            "grid": g}


def main() -> None:
    recs = sys.argv[1:] or list(RECS)
    for name in recs:
        frames = load(ROOT / "navmesh_recordings" / name)
        if not frames:
            print(f"\n=== {name} === 没有关键帧")
            continue
        print(f"\n=== {name} ===  {len(frames)} 关键帧")
        print(f"  {'q_tiers':>8s} {'ray_clear':>10s} {'OCC':>7s} {'FREE':>7s} {'UNK':>7s} "
              f"{'走过∩OCC':>10s} {'走过格数':>9s} {'误判率':>8s} {'看穿清零':>9s}")
        base = None
        for qt, rc in COMBOS:
            c = MapperConfig()
            c.q_tiers, c.ray_clear = qt, rc
            r = one(frames, c)
            ow = r["occ_walked"]
            # 误判率 = 走过的格里有几成被判障碍。走过总格数为分母（含 FREE/OCC/UNK）。
            rate = ow / max(r["line"], 1) * 100
            print(f"  {str(qt):>8s} {str(rc):>10s} {r['occ']:7d} {r['free']:7d} {r['unk']:7d} "
                  f"{ow:10d} {r['line']:9d} {rate:7.2f}% {r['ray_cleared']:9d}")
            if (qt, rc) == (True, True):
                base = r
        if base:
            print(f"  基线（两套都开）走过∩OCC = {base['occ_walked']} 格；"
                  f"关掉 ray_clear 后 OCC 从 {base['occ']} 变到上表对应行。")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
