# -*- coding: utf-8 -*-
"""这三段录制里到底有没有"低矮区域"？——2 m 以下的水平面筛查。

    python -m tools.low_ceiling_probe [录制名...]

## 要回答的

上一轮结论是净空过滤在现有录制里滤掉 0 个格，于是问题变成：
**是过滤没用，还是这些世界根本没有低矮区域？**

如果"低矮区域"存在，2 m 以下必须有一批**水平**面（天花板/门楣/桥底/车顶）。
竖直的墙不算——墙已经在 2D 栅格里被当成障碍处理了。

## 怎么区分水平面和竖直墙

对 0.3~2.0 m 带的每格取众数高度 ``mode_h``，再做 3×3 邻域高度极差：

    极差小  ⇒ 邻域是一个水平面  ⇒ 矮天花板
    极差大  ⇒ 邻域高度乱跳      ⇒ 墙/杂物/噪声

同一个判据在 ``band_levels`` 里量过 3.85 m 那层：残差 0.00 m、连贯 92%，
而随机置换对照是 0.10 m。这次用同样的阈值往下压到 2 m 以下。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import MapperConfig  # noqa: E402
from tools import band_levels                            # noqa: E402
from tools.band_levels import CELL_M, HBIN, MIN_PTS       # noqa: E402
from tools.precision_probe import load                    # noqa: E402

RECS = ("20261001_044153", "20260929_045615", "20261001_044120")

# 水平面判据：3×3 邻域众数高度极差小于这个值，就认为是一张平的水平面
PLANAR_TOL = 0.15
# 邻域里至少要有这么多个有效格才敢下结论（否则可能是数据边界的巧合）
MIN_NB = 6


def low_band_mask(h: np.ndarray, c: MapperConfig, lo_i: int) -> np.ndarray:
    """只用到 lo_i=0：0.3~2.0 m 这一条带（地面容差之上、障碍带上沿之下）。"""
    return (h > c.ground_tol_m) & (h <= c.obst_top_m)


def neighbour_spread(mode_h: np.ndarray, valid: np.ndarray):
    """3×3 邻域内 mode_h 的极差，以及窗口内的有效格数。"""
    M = np.where(valid, mode_h, np.nan)
    H, W = M.shape
    stack = np.full((9, H, W), np.nan)
    for k, (dy, dx) in enumerate(
            [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]):
        ys0, ys1 = max(0, dy), H + min(0, dy)
        xs0, xs1 = max(0, dx), W + min(0, dx)
        stack[k, ys0:ys1, xs0:xs1] = M[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    with np.errstate(invalid="ignore"):
        hi = np.nanmax(stack, axis=0)
        lo = np.nanmin(stack, axis=0)
    n_ok = np.sum(~np.isnan(stack), axis=0)
    return hi - lo, n_ok


def probe(frames, c: MapperConfig):
    H, _ny, _nx, lo = band_levels._abs_hist(frames, 0)
    n = H.sum(axis=0)
    valid = n >= MIN_PTS
    if not valid.any():
        return None
    mode = H.argmax(axis=0)
    mode_h = np.where(valid, lo + mode * HBIN, np.nan)

    spread, n_ok = neighbour_spread(mode_h, valid)
    planar = valid & (n_ok >= MIN_NB) & (spread <= PLANAR_TOL)
    return mode_h, valid, spread, n_ok, planar


def main() -> None:
    recs = sys.argv[1:] or list(RECS)
    for name in recs:
        frames = load(ROOT / "navmesh_recordings" / name)
        if not frames:
            print(f"\n=== {name} ===\n  没有关键帧")
            continue
        r = probe(frames, MapperConfig())
        print(f"\n=== {name} ===  {len(frames)} 关键帧  统计格 {CELL_M} m")
        if r is None:
            print("  0.3~2.0 m 带里没有一格攒够点数")
            continue
        mode_h, valid, spread, n_ok, planar = r
        print(f"  0.3~2.0 m 带 有点数 {int(valid.sum())} 格 / 全图 {mode_h.size} 格 "
              f"({valid.sum() / mode_h.size * 100:.1f}%)")
        print(f"  其中判为水平面（3×3 极差 ≤ {PLANAR_TOL} m）{int(planar.sum())} 格 "
              f"({planar.sum() / max(int(valid.sum()), 1) * 100:.1f}% of 有点数)")
        if not planar.any():
            print("  ⇒ 这段录制里【没有低矮区域】，净空过滤无对象可滤")
            continue
        v = mode_h[planar]
        print(f"  这些水平面的高度  min {v.min():.2f}  p5 {np.percentile(v, 5):.2f}  "
              f"中位 {np.median(v):.2f}  p95 {np.percentile(v, 95):.2f}  max {v.max():.2f} m")
        # 直方图：矮于 1.8 m（几乎所有 avatar 都过不去）的有多少
        for lim in (1.5, 1.8, 2.0):
            k = int((v <= lim).sum())
            print(f"    ≤ {lim:.1f} m: {k:6d} 格 ({k / len(v) * 100:5.1f}%)")
        # 对照：随机置换后的极差，确认判据不是把噪声也判成平面
        rs = np.random.default_rng(7)
        flat = mode_h[valid]
        perm = rs.permutation(flat)
        pm = np.full(mode_h.shape, np.nan)
        pm[valid] = perm.reshape(valid.sum())
        ps, pn = neighbour_spread(pm, valid)
        pp = valid & (pn >= MIN_NB) & (ps <= PLANAR_TOL)
        print(f"    对照：随机打乱后同样判为水平面的 {int(pp.sum())} 格 "
              f"(真值 {int(planar.sum())} 格，判据信噪比 "
              f"{planar.sum() / max(int(pp.sum()), 1):.1f}×)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    band_levels._band_mask = low_band_mask
    main()
