# -*- coding: utf-8 -*-
r"""`pos_cell_4m` 在独立几何裁判上跑赢 T_map（2.46 vs 2.98 m）—— 可疑，查是不是"更挤"。

按 cell 分块仿射有个**构造性副作用**：落在同一 cell 的 DR 点被**同一个仿射**处理。
而重访点在 T_map 下位置接近 ⇒ 必然落进同一 cell ⇒ 它们的 DR 位置（漂移后可能相距很远）
被同一个仿射压到相近处。**这会把"重合"伪装成"准确"。**

两刀揭开：
1. **轨迹长度保持率**：真实步长是已知的（kf 按 kf_dist 触发）。形变若把步长压小 ⇒ 在压扁轨迹。
2. **cell 边界跳变**：相邻关键帧跨 cell 时，形变后的间距会出现不连续 ⇒ 地图被切碎。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # research/tools/ → 仓库根
sys.path.insert(0, str(ROOT))


def load(sid: str):
    z = np.load(ROOT / "navmesh_memory" / "wrld_home-7cf435ea" / "sessions" / sid / "poses.npz")
    d = np.asarray(z["dist_m"], float)
    o = np.argsort(d)
    return (d[o], np.asarray(z["T_map"], np.float64)[o][:, :2, 3],
            np.asarray(z["T_dr"], np.float64)[o][:, :2, 3])


def grid_bucket(P, cell):
    return (np.floor(P[:, 0] / cell).astype(int) * 1000 + np.floor(P[:, 1] / cell).astype(int))


for sid in (sys.argv[1:] or ["20261005_235237", "20261001_044153"]):
    d, Pm, Pd = load(sid)
    n = len(d)
    print(f"\n{'='*72}\n会话 {sid} · {n} kf · 里程 {d.min():.0f}→{d.max():.0f} m（OSC 路程口径）")

    def step(P):
        return np.linalg.norm(np.diff(P, axis=0), axis=1)

    s_dr, s_map = step(Pd), step(Pm)
    print(f"\n【刀1】相邻关键帧步长（真实路程由 OSC 给出：Δdist 中位 {np.median(np.diff(d)):.3f} m）")
    print(f"  T_dr  步长中位 {np.median(s_dr):.3f} m · T_map {np.median(s_map):.3f} m"
          f" ⇒ 回路环后步长保持率 {np.median(s_map)/np.median(s_dr)*100:.1f}%")

    for cell in (4.0, 8.0):
        bk = grid_bucket(Pd, cell)
        out = np.empty_like(Pd)
        for b in np.unique(bk):
            m = bk == b
            if m.sum() >= 6:
                X = np.column_stack([Pd[m], np.ones(m.sum())])
                s, *_ = np.linalg.lstsq(X, Pm[m], rcond=None)
                out[m] = Pd[m] @ s[:2, :2].T + s[2]
            else:
                out[m] = Pm[m]
        s_cell = step(out)
        cross = bk[:-1] != bk[1:]                 # 跨 cell 边界的相邻对
        jump = s_cell[cross]
        inside = s_cell[~cross]
        print(f"\n  pos_cell cell={cell:.0f}m（{len(np.unique(bk))} 块）")
        print(f"    步长中位 {np.median(s_cell):.3f} m ⇒ 保持率 {np.median(s_cell)/np.median(s_dr)*100:5.1f}%"
              f"   {'⚠️ 轨迹被压扁' if np.median(s_cell)/np.median(s_dr) < 0.9 else ''}")
        print(f"    块内相邻步长中位 {np.median(inside):.3f} m · **跨块**步长中位 {np.median(jump):.3f} m"
              f" ⇒ 跨块跳变 ×{np.median(jump)/max(np.median(inside),1e-9):.2f}")
        print(f"    总长度：DR {s_dr.sum():.0f} m · T_map {s_map.sum():.0f} m · pos_cell {s_cell.sum():.0f} m"
              f" · OSC 真值 {d.max()-d.min():.0f} m")
