# -*- coding: utf-8 -*-
"""``ray_near_exempt`` 的 A/B——动的是 ray_clear 本体，两把尺子必须同时看。

    python -m tools.ray_exempt_ab [录制名...]

## 为什么这一刀特别危险

``ray_clear`` 是整套虚假障碍修复的主力：实测（tools/ray_clear_ab.py，044153）
关掉它，用户走过的路径有 **12.97%** 被判障碍；开着是 **1.35%**。
而尺子 1 就是"走过∩OCC 率"——**它恰好就是这个机制的验收指标**。

所以尺子 1 一变好就该警惕：这一刀可能只是在把尺子 1 刷好看，代价全记在尺子 2 上。
两把同时看，交换率才是价格。

## 机制上的疑问（这一刀想验证的）

看穿判据要求整列**所有** z 层都不通过才清零。一根从相机掠过 0.85 m 矮墙顶、
往墙后地面俯冲的射线，会在墙顶那一层 0.80~0.90 m 处穿过去记一次"看穿"。
实测该层 349 个有打中的格**全部**被看穿票压过 ⇒ 真墙被判成可走。
豁免有近距票的格，理论上应该正好放回这批，且不碰纯远场的假障碍。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import KeyframeGridMapper, MapperConfig   # noqa: E402
from backend.nav_grid import OCC                                   # noqa: E402
from tools.precision_probe import load                             # noqa: E402
from tools.nav_audit import (MIN_HIT_LAYERS, MIN_NB, MIN_PTS, PLANAR_TOL,  # noqa: E402
                             band_pass, _spread)

RECS = ("20261001_044153", "20260929_045615")


def walked_line(ng, walked) -> np.ndarray:
    line = np.zeros(ng.grid.shape, np.uint8)
    for poly in walked:
        pts = np.array([ng.to_cell(p)[::-1] for p in np.asarray(poly, float)], np.int32)
        if len(pts) >= 2:
            cv2.polylines(line, [pts.reshape(-1, 1, 2)], False, 1, 1)
    return line


def run(frames, exempt: bool):
    c = MapperConfig()
    c.ray_near_exempt = exempt
    m = KeyframeGridMapper(c)
    for i, fr in enumerate(frames):
        m.add_keyframe(i, fr[1], fr[2])
    ng = m.rasterize()
    ng.build(radius_m=0.25, walked=m.walked())
    g = ng.grid

    line = walked_line(ng, m.walked())
    nl = int((line > 0).sum())
    blocked = int(((g == OCC) & (line > 0)).sum())
    r1 = blocked / max(nl, 1) * 100

    hp = band_pass(m, c, len(frames))
    if hp is None:
        return r1, 0, int((g == OCC).sum()), 0, nl, blocked
    mode_h, n_pts, mean_r = hp
    g_mode = m._cut_acc(np.nan_to_num(mode_h, nan=0.0))
    g_n = m._cut_acc(n_pts)
    g_r = m._cut_acc(np.nan_to_num(mean_r, nan=0.0))
    valid = g_n >= MIN_PTS
    spread, n_ok = _spread(g_mode, valid)
    planar = valid & (n_ok >= MIN_NB) & (spread <= PLANAR_TOL)
    hz = np.stack([m._cut_acc(m._ray_hit[z]) for z in range(len(m._ray_hit))])
    cand = planar & (g_r < c.q_near_m) & ((hz > 0).sum(axis=0) >= MIN_HIT_LAYERS)
    idx = np.flatnonzero(cand.ravel())
    leak = int((g.ravel()[idx] != OCC).sum())
    return r1, leak, int((g == OCC).sum()), int(cand.sum()), nl, blocked


def main() -> None:
    for name in (sys.argv[1:] or list(RECS)):
        frames = load(ROOT / "navmesh_recordings" / name)
        if not frames:
            print(f"\n=== {name} === 没有关键帧")
            continue
        print(f"\n=== {name} ===  {len(frames)} 关键帧")
        print(f"  {'豁免':>6s} {'OCC':>7s} {'尺子1 走过∩OCC':>15s} {'尺子2 漏放':>11s} "
              f"{'候选格':>7s} {'Δ漏放':>7s} {'Δ路径格':>9s}  真实交换比")
        base = None
        for v in (False, True):
            r1, leak, occ, cand, nl, blocked = run(frames, v)
            if base is None:
                base = (leak, blocked)
                d2, d1 = 0, 0
                rate = "（基线 = 不豁免）"
            else:
                d2 = int(base[0] - leak)      # 回收的真结构格
                d1 = int(blocked - base[1])   # 新封死的路径格
                rate = f"{d2 / d1:.1f} : 1 划算" if d1 > 0 and d2 > 0 else (
                    "∞（路径零代价）" if d1 == 0 and d2 > 0 else "—")
            print(f"  {str(v):>6s} {occ:7d} {r1:14.2f}% {leak:11d} {cand:7d} "
                  f"{d2:7d} {d1:9d}  {rate}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
