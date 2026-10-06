# -*- coding: utf-8 -*-
r"""独立几何验收：三种形变到底有没有**真的**改善几何？

上一轮（`_falsify_deform.py`）已证：按里程分段是位置的**多值函数** ⇒ 撕裂地图，出局。
但"按位置分块"是几何合法的（单值），且把对 T_map 的拟合残差降了不少
（235237：identity 3.97 → global 2.54 → 位置分块 0.78 m）。

那个残差**不足以**下结论 —— 它的真值就是 T_map，而 T_map 是回环产物（循环）。
本工具换一个**独立裁判**：

    取**重访点对**（物理上是同一地点、里程差很大），用各位姿源把稠密点云投到世界系，
    量两片云的**最近邻距离**。位姿越准 ⇒ 同一面墙两次观测重合得越好。
    这个量是纯几何的，不拿 T_map 当真值（T_map 只在**选样本**时用，评估一视同仁）。

形变拟合**留出该对**：拟合时排除待评估的 2 个关键帧 ⇒ 测的是泛化，不是记忆。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]   # research/tools/ → 仓库根
sys.path.insert(0, str(ROOT))

SID = sys.argv[1] if len(sys.argv) > 1 else "20261005_235237"
def _pick_rec(sid: str) -> Path:
    """会话 id 与录制目录名常常差几秒（如会话 235237 ↔ 录制 235232）⇒ 取同日最近的一个。"""
    cand = sorted(p for p in (ROOT / "navmesh_recordings").iterdir()
                  if p.is_dir() and (p / "kf").is_dir() and p.name[:8] == sid[:8])
    if not cand:
        raise SystemExit(f"找不到 {sid} 对应的录制目录")
    return min(cand, key=lambda p: abs(int(p.name[9:]) - int(sid[9:])))


REC = _pick_rec(SID)
NPER = 3000
MAXPAIR = 24

z = np.load(ROOT / "navmesh_memory" / "wrld_home-7cf435ea" / "sessions" / SID / "poses.npz")
ids = np.asarray(z["ids"])
d = np.asarray(z["dist_m"], float)
T_map = np.asarray(z["T_map"], np.float64)
T_dr = np.asarray(z["T_dr"], np.float64)
o = np.argsort(d)
ids, d, T_map, T_dr = ids[o], d[o], T_map[o], T_dr[o]
P_map, P_dr = T_map[:, :2, 3], T_dr[:, :2, 3]
n = len(ids)
print(f"会话 {SID} · {n} kf · 里程 {d.min():.0f}→{d.max():.0f} m")

# ---------- 选重访对（用 T_map 位置：它是现有最好的位置估计，只用于挑样本） ----------
tree = cKDTree(P_map)
pr = np.array(tree.query_pairs(1.0, output_type="ndarray"))
far = np.abs(d[pr[:, 0]] - d[pr[:, 1]]) > 30.0
pr = pr[far]
rng = np.random.default_rng(7)
if len(pr) > MAXPAIR:
    pr = pr[rng.choice(len(pr), MAXPAIR, replace=False)]
print(f"重访对（T_map 位置 <1.0 m 且里程差 >30 m）：{len(pr)} 对"
      f" · 里程差中位 {np.median(np.abs(d[pr[:,0]]-d[pr[:,1]])):.0f} m\n")


def grid_bucket(P: np.ndarray, cell: float = 4.0) -> np.ndarray:
    return (np.floor(P[:, 0] / cell).astype(int) * 1000 + np.floor(P[:, 1] / cell).astype(int))


def build_deform(exclude: set[int]) -> dict:
    """在排除 exclude 的关键帧上拟合形变场。返回 {bucket: (A, b)} + 全局 (A, b)。"""
    m = np.ones(n, bool)
    m[list(exclude)] = False
    X = np.column_stack([P_dr[m], np.ones(m.sum())])
    sol, *_ = np.linalg.lstsq(X, P_map[m], rcond=None)
    glob = (sol[:2, :2].T.copy(), sol[2].copy())
    cells: dict = {}
    bk = grid_bucket(P_dr)
    for b in np.unique(bk[m]):
        mb = m & (bk == b)
        if mb.sum() >= 6:
            Xb = np.column_stack([P_dr[mb], np.ones(mb.sum())])
            s, *_ = np.linalg.lstsq(Xb, P_map[mb], rcond=None)
            cells[b] = (s[:2, :2].T.copy(), s[2].copy())
    return {"glob": glob, "cells": cells, "bk": bk}


def pose_variants(i: int, df: dict) -> dict[str, np.ndarray]:
    """返回第 i 帧在各形变下的**完整 4x4 位姿**（只改平移，yaw 实测恒等）。"""
    A, b = df["glob"]
    p_glob = A @ P_dr[i] + b
    bk = df["bk"][i]
    p_cell = (df["cells"][bk][0] @ P_dr[i] + df["cells"][bk][1]) if bk in df["cells"] else p_glob
    out = {}
    for nm, p in (("identity(T_dr)", P_dr[i]), ("global_affine", p_glob),
                  ("pos_cell_4m", p_cell), ("T_map(回环产物)", P_map[i])):
        T = T_dr[i].copy()
        T[:2, 3] = p
        out[nm] = T
    return out


def cloud(i: int, T: np.ndarray) -> np.ndarray:
    f = REC / "kf" / f"{ids[i]:06d}.npz"
    with np.load(f) as kf:
        p = np.asarray(kf["pts"], np.float64)
    if len(p) > NPER:
        p = p[rng.choice(len(p), NPER, replace=False)]
    return p @ T[:3, :3].T + T[:3, 3]


names = ["identity(T_dr)", "global_affine", "pos_cell_4m", "T_map(回环产物)"]
acc = {k: [] for k in names}
for a, b in pr:
    df = build_deform({a, b})                       # ★ 留出：拟合时排除这两帧
    vs_a, vs_b = pose_variants(a, df), pose_variants(b, df)
    for nm in names:
        ca, cb = cloud(a, vs_a[nm]), cloud(b, vs_b[nm])
        dist_a = cKDTree(cb).query(ca, workers=-1)[0]
        dist_b = cKDTree(ca).query(cb, workers=-1)[0]
        acc[nm].append(np.median(np.concatenate([dist_a, dist_b])))

print("重访处两片稠密云的**最近邻距离中位**（越小 = 同一面墙两次观测重合越好）")
print(f"{'位姿源':<20}{'中位(m)':>10}{'p25':>10}{'p75':>10}   相对 identity")
base = float(np.median(acc["identity(T_dr)"]))
for nm in names:
    v = np.array(acc[nm])
    print(f"{nm:<20}{np.median(v):10.3f}{np.percentile(v,25):10.3f}{np.percentile(v,75):10.3f}"
          f"   {100*(1-np.median(v)/base):+7.1f}%")
