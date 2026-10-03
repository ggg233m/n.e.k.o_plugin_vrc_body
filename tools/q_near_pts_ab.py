# -*- coding: utf-8 -*-
"""``q_near_pts`` 的 A/B——两把尺子必须同时看。

    python -m tools.q_near_pts_ab [录制名...]

## 为什么非看两把不可

    尺子 1  走过∩OCC 率          放宽近带会**变差**（路径上多出障碍）
    尺子 2  漏放格（近距平面证据） 放宽近带应该**变好**（真结构回来）

只报尺子 1 会得到"别改"，只报尺子 2 会得到"全开"。判据是两者的**交换率**：
尺子 2 少漏放 N 格，尺子 1 恶化 M 格，N/M 就是这笔交易的价格。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import KeyframeGridMapper, MapperConfig, _ground_correction  # noqa: E402
from backend.nav_grid import FREE, OCC, UNK                                          # noqa: E402
from tools.precision_probe import load                                              # noqa: E402
from tools.nav_audit import HBIN, MIN_PTS, MIN_NB, MIN_HIT_LAYERS, PLANAR_TOL, band_pass, _spread  # noqa: E402

VALUES = (0, 2, 1)
RECS = ("20261001_044153", "20260929_045615")


def walked_line(ng, walked) -> np.ndarray:
    line = np.zeros(ng.grid.shape, np.uint8)
    for poly in walked:
        pts = np.array([ng.to_cell(p)[::-1] for p in np.asarray(poly, float)], np.int32)
        if len(pts) >= 2:
            cv2.polylines(line, [pts.reshape(-1, 1, 2)], False, 1, 1)
    return line


def run(frames, q_near_pts: int):
    c = MapperConfig()
    c.q_near_pts = q_near_pts
    m = KeyframeGridMapper(c)
    for i, fr in enumerate(frames):
        m.add_keyframe(i, fr[1], fr[2])
    ng = m.rasterize()
    ng.build(radius_m=0.25, walked=m.walked())
    g = ng.grid

    line = walked_line(ng, m.walked())
    nl = int((line > 0).sum())
    r1 = int(((g == OCC) & (line > 0)).sum()) / max(nl, 1) * 100

    hp = band_pass(m, c, len(frames))
    if hp is None:
        return r1, 0, int((g == OCC).sum()), 0
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
    st = g.ravel()[idx]
    leak = int((st != OCC).sum())
    return r1, leak, int((g == OCC).sum()), int(cand.sum())


def main() -> None:
    for name in (sys.argv[1:] or list(RECS)):
        frames = load(ROOT / "navmesh_recordings" / name)
        if not frames:
            print(f"\n=== {name} === 没有关键帧")
            continue
        print(f"\n=== {name} ===  {len(frames)} 关键帧")
        print(f"  {'q_near_pts':>10s} {'OCC':>7s} {'尺子1 走过∩OCC':>15s} {'尺子2 漏放':>11s} "
              f"{'候选格':>7s} {'Δ尺子2':>8s} {'Δ尺子1(pp)':>11s}")
        base = None
        for v in VALUES:
            r1, leak, occ, cand = run(frames, v)
            if base is None:
                base = (r1, leak)
                d2, d1 = 0, 0.0
            else:
                d2 = int(base[1] - leak)       # 正数 = 少漏放，好
                d1 = r1 - base[0]             # 正数 = 路径误判变差，坏
            tag = "（基线 = 关闭）" if v == 0 else (f"Δ漏放 {d2:+d}  Δ路径 {d1:+.2f} pp"
                                                  f"  交换率 {(-d2 / d1) if d1 > 0e-9 else float('inf'):.1f}×")
            print(f"  {v:>10d} {occ:7d} {r1:14.2f}% {leak:11d} {cand:7d} {d2:8d} "
                  f"{d1:11.2f}  {tag}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
