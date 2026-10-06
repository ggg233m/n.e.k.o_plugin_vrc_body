# -*- coding: utf-8 -*-
r"""证伪自己的实验：分段形变那个"改善 100%"是不是真的？

三刀，刀刀指向**循环论证**与**几何多值**：

1. **末段冻结偏移**：文档 §9.3 说末段 ~1/4 会话没被回环碰过 ⇒ `T_map - T_dr` 是常量。
   拟合它 = 记住一个常量，残差 0.000 **不代表任何几何精度**。
   判据：末段修正量的段内方差 ≈ 0？
2. **几何多值**：修正量是**里程**的函数，但同一地点会在不同里程被重访。
   若重访点的修正量不同 ⇒ 按里程的形变是**位置的多值函数** ⇒ 会把同一地点
   映射到两个位置（撕裂地图），无论残差多小都必须出局。
3. **把 T_map 当真值本身就是循环**：T_map 是回环产物，它的误差结构正是我们要修的东西。
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
    return (d[o], np.asarray(z["T_map"], float)[o][:, :2, 3],
            np.asarray(z["T_dr"], float)[o][:, :2, 3])


for sid in (sys.argv[1:] or ["20261001_044153", "20261005_235237"]):
    d, Pm, Pd = load(sid)
    n = len(d)
    r = Pm - Pd                                   # 需要的修正量
    rn = np.linalg.norm(r, axis=1)
    print(f"\n{'='*76}\n会话 {sid} · {n} kf · 里程 {d.min():.1f}→{d.max():.1f} m")
    print(f"修正量 |r|：中位 {np.median(rn):.3f} · 最大 {rn.max():.3f} m\n")

    # ---------- 刀 1：末段是不是冻结常量 ----------
    print("【刀1】末段修正量是否冻结为常量？（是 ⇒ 拟合它=记常量，残差0也无意义）")
    K = 8
    e = np.quantile(d, np.linspace(0, 1, K + 1))
    tail = d >= e[-2]
    rt = r[tail]
    spread = np.linalg.norm(rt - rt.mean(0), axis=1)
    print(f"  末段（里程 ≥{e[-2]:.1f} m，{tail.sum()} 帧）修正量："
          f"均值向量 {np.round(rt.mean(0),3)} · 段内散布 中位 {np.median(spread):.4f} m · max {spread.max():.4f} m")
    print(f"  ⇒ {'**冻结常量**（段内散布≈0）——拟合它毫无几何价值' if np.median(spread) < 0.02 else '末段仍有变化，不是常量'}")

    # ---------- 刀 2：几何多值 ----------
    print("\n【刀2】同一地点在不同里程被重访时，修正量一致吗？")
    print("      （不一致 ⇒ 按里程的形变是位置的多值函数 ⇒ 撕裂地图，必须出局）")
    from scipy.spatial import cKDTree
    tree = cKDTree(Pd)
    pairs = tree.query_pairs(0.6, output_type="ndarray")     # DR 位置相距 <0.6 m 的对
    far = np.abs(d[pairs[:, 0]] - d[pairs[:, 1]]) > 20.0     # 且里程差 >20 m ⇒ 真重访
    pr = pairs[far]
    print(f"  重访对（DR 位置 <0.6 m 且里程差 >20 m）：{len(pr)} 对")
    if len(pr):
        dr_corr = np.linalg.norm(r[pr[:, 0]] - r[pr[:, 1]], axis=1)
        dm = np.abs(d[pr[:, 0]] - d[pr[:, 1]])
        print(f"  它们的修正量之差：中位 {np.median(dr_corr):.3f} m · p90 {np.percentile(dr_corr,90):.3f} m"
              f" · max {dr_corr.max():.3f} m   （里程差中位 {np.median(dm):.0f} m）")
        print(f"  参照：修正量本身中位 {np.median(rn):.3f} m")
        verdict = ("**多值**：同一个位置需要相差很大的修正 ⇒ 按里程分段会把同一地点映射到不同位置"
                   if np.median(dr_corr) > 0.3 else "重访处修正量一致 ⇒ 形变可视为位置的函数")
        print(f"  ⇒ {verdict}")

    # ---------- 刀 3：对照——把"里程"换成"位置"能否降残差 ----------
    print("\n【刀3】对照：形变若按**位置**分块（而非里程），留出残差如何？")
    print("      （若位置分块明显劣于里程分块 ⇒ 进一步证明抓的是时间累积，不是几何）")
    from scipy.spatial import cKDTree as KT
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]

    def fit_eval(keys_tr, keys_te, bucket_te):
        """按 bucket 分块，块内仿射；测试块用训练集同块参数。"""
        out = np.empty((len(te), 2))
        for b in np.unique(bucket_te):
            mt, me = keys_tr == b, bucket_te == b
            if not me.any():
                continue
            if mt.sum() >= 3:
                X = np.column_stack([Pd[tr][mt], np.ones(mt.sum())])
                sol, *_ = np.linalg.lstsq(X, Pm[tr][mt], rcond=None)
                A, bb = sol[:2, :2].T, sol[2]
            else:                                    # 该块无训练数据 ⇒ 用全局
                X = np.column_stack([Pd[tr], np.ones(len(tr))])
                sol, *_ = np.linalg.lstsq(X, Pm[tr], rcond=None)
                A, bb = sol[:2, :2].T, sol[2]
            out[me] = Pd[te][me] @ A.T + bb
        return np.linalg.norm(out - Pm[te], axis=1)

    # 里程分块（K=16）
    eK = np.quantile(d, np.linspace(0, 1, 17))
    b_mile_tr = np.clip(np.digitize(d[tr], eK[1:-1]), 0, 15)
    b_mile_te = np.clip(np.digitize(d[te], eK[1:-1]), 0, 15)
    err_mile = fit_eval(b_mile_tr, b_mile_te, b_mile_te)

    # 位置分块（2D 网格 4 m）
    def grid_bucket(P):
        return (np.floor(P[:, 0] / 4.0).astype(int) * 1000 + np.floor(P[:, 1] / 4.0).astype(int))
    b_pos_tr, b_pos_te = grid_bucket(Pd[tr]), grid_bucket(Pd[te])
    err_pos = fit_eval(b_pos_tr, b_pos_te, b_pos_te)

    # 基线：全局仿射
    X = np.column_stack([Pd[tr], np.ones(len(tr))])
    sol, *_ = np.linalg.lstsq(X, Pm[tr], rcond=None)
    err_glob = np.linalg.norm(Pd[te] @ sol[:2, :2].T + sol[2] - Pm[te], axis=1)
    err_id = np.linalg.norm(Pd[te] - Pm[te], axis=1)

    for nm, err in (("identity", err_id), ("global_affine", err_glob),
                    ("按里程分块 K=16", err_mile), ("按位置分块 4m 网格", err_pos)):
        print(f"  {nm:<22} 中位 {np.median(err):6.3f} · p90 {np.percentile(err,90):6.3f}")
