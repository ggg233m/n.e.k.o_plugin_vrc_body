# -*- coding: utf-8 -*-
"""几何门为什么漏？——把 045615 那批漏放格逐条归因到 ``_near_plane`` 的某一项。

    python -m tools.plane_gate_why [录制名...]

## 要回答的

几何门（``ray_near_exempt`` + ``_near_plane``）两段都净赚，但**回收率差很远**：
044153 收回 135/202（67%），045615 只收回 47/193（24%）。

这个差距本身说明门还有没处理的东西，**比调阈值重要**——现在只有两个数据点，
动 ``ray_plane_pts``/``ray_plane_tol`` 就是在两点之间过拟合。

## 怎么归因

候选集用**线下那套已验过的判据**（直方图众数高度 + 真实观测距离 + ≥2 个打中层，
见 ``nav_audit`` 的尺子 2），热路径那套（矩推的每格平均高度 + 近距票占比）是它的近似。
对每个候选格，报告它**过没过热路径的哪一项**：

    pts     n_o >= ray_plane_pts
    near    近距票占比 >= ray_plane_near
    inside  3×3 有效邻居 >= 6（图沿，采样边界）
    flat    3×3 平均高度极差 <= ray_plane_tol

过了 = 被回收；没过 = 仍然漏放，按"最先失败的那一项"分组。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import KeyframeGridMapper, MapperConfig   # noqa: E402
from backend.nav_grid import OCC                                   # noqa: E402
from tools.precision_probe import load                             # noqa: E402
from tools.nav_audit import (MIN_HIT_LAYERS, MIN_NB, MIN_PTS, PLANAR_TOL,  # noqa: E402
                             band_pass, _spread)

RECS = ("20261001_044153", "20260929_045615")
ORDER = ("pts", "near", "inside", "flat")


def main() -> None:
    for name in (sys.argv[1:] or list(RECS)):
        frames = load(ROOT / "navmesh_recordings" / name)
        if not frames:
            print(f"\n=== {name} === 没有关键帧")
            continue
        c = MapperConfig()
        c.ray_near_exempt = True
        m = KeyframeGridMapper(c)
        for i, fr in enumerate(frames):
            m.add_keyframe(i, fr[1], fr[2])
        ng = m.rasterize()
        g = ng.grid

        # 候选集：线下那套已验判据（尺子 2），裁到 NavGrid 行序
        mode_h, n_pts, mean_r = band_pass(m, c, len(frames))
        g_mode = m._cut_acc(np.nan_to_num(mode_h, nan=0.0))
        g_n = m._cut_acc(n_pts)
        g_r = m._cut_acc(np.nan_to_num(mean_r, nan=0.0))
        spread, n_ok = _spread(g_mode, g_n >= MIN_PTS)
        planar = (g_n >= MIN_PTS) & (n_ok >= MIN_NB) & (spread <= PLANAR_TOL)
        hz = np.stack([m._cut_acc(m._ray_hit[z]) for z in range(len(m._ray_hit))])
        cand = planar & (g_r < c.q_near_m) & ((hz > 0).sum(axis=0) >= MIN_HIT_LAYERS)

        # 热路径那套（累加器行序 → 裁到 NavGrid 行序）
        p = m._near_plane_parts()
        gate = {k: m._cut_acc(v.astype(np.float32)) > 0.5 for k, v in p.items()}

        idx = np.flatnonzero(cand.ravel())
        st = g.ravel()[idx]
        still = st != OCC                                   # 豁免后仍然是漏放
        caught = ~still
        print(f"\n=== {name} ===  {len(frames)} 关键帧   候选 {len(idx)} 格")
        print(f"  几何门回收 {int(caught.sum()):4d}   仍漏放 {int(still.sum()):4d}"
              f"   回收率 {caught.sum() / max(len(idx), 1) * 100:.0f}%")

        # 先看这批格本身长什么样
        for k in ("pts", "near", "inside", "flat"):
            r = gate[k].ravel()[idx]
            print(f"    {k:7s} 通过  回收组 {int(r[caught].sum()):4d}/{int(caught.sum()):4d}   "
                  f"漏放组 {int(r[still].sum()):4d}/{int(still.sum()):4d}")

        # 漏放组：按"最先失败的那一项"分组
        miss = idx[still]
        if len(miss):
            why = np.full(len(miss), "全过(但仍不是障碍)", dtype=object)
            todo = np.ones(len(miss), bool)
            for k in ORDER:
                hit = ~gate[k].ravel()[miss] & todo
                why[hit] = k
                todo &= ~hit
            print("    漏放组归因（最先失败的那一项）：")
            for k in list(ORDER) + ["全过(但仍不是障碍)"]:
                n = int((why == k).sum())
                if n:
                    print(f"      {k:20s} {n:4d} 格 ({n / len(miss) * 100:5.1f}%)")
            # 近距票占比与观测距离的实情——这两项最可能是瓶颈
            n_o = m._cut_acc(m._acc_o).ravel()[miss]
            on = m._cut_acc(m._acc_on).ravel()[miss]
            frac = on / np.maximum(n_o, 1e-9)
            print(f"      近距票占比  p25 {np.percentile(frac, 25):.2f}  中位 "
                  f"{np.median(frac):.2f}  p75 {np.percentile(frac, 75):.2f}   "
                  f"（门 {c.ray_plane_near}）")
            print(f"      n_o        p25 {np.percentile(n_o, 25):.0f}  中位 "
                  f"{np.median(n_o):.0f}  p75 {np.percentile(n_o, 75):.0f}   "
                  f"（门 {c.ray_plane_pts}）")
            rr = g_r.ravel()[miss]
            print(f"      观测距离   p25 {np.percentile(rr, 25):.2f}  中位 "
                  f"{np.median(rr):.2f}  p75 {np.percentile(rr, 75):.2f} m   "
                  f"（候选集已要求 < {c.q_near_m}）")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
