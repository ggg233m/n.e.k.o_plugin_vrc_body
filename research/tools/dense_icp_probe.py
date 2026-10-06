# -*- coding: utf-8 -*-
r"""可行性探针：**稠密点云配准（ICP）能不能当独立位姿约束源？**

为什么问这个
------------
稀疏路已被证伪/封死：
* 回环靠 ORB+PnP，`min_inliers=50` ⇒ 低纹理世界 `few_matches` 拒了绝大多数（044153 当场 318 次）；
* 尺度 `loop_selfcal` 的 s 逐场 1.12/1.73/1.35/2.14、IQR 跨度 1.5–2.7 倍，且文档明确
  "**在拿到独立几何锚点之前不建议直接改**"（`尺度标定门依赖-9%分歧（2026-09-24）.md:117`）。

而**稠密点云是被闲置的资产**：每关键帧 4.4 万个双目点，位姿优化从来没用过它们。
双目是**绝对尺度**（基线 0.126、fx 已知）⇒ 稠密配准天然提供米制约束，
不依赖 OSC、不受"静止/匀速不发包"影响。

要判的事
--------
ICP 的**收敛盆地**有多大：初值误差 δ 多大时还能收敛到正确相对位姿？
* 盆地 ≳ 回环修正后的残余误差（<1 m）⇒ 可接在现有链路后面，作为独立约束；
* 盆地很小（<0.3 m）⇒ 只能当精修，救不了长距离漂移。

方法：真值相对位姿取 `T_map` 的相对变换（回环产物，是本场最好的估计），
人为加扰动 δ 当 ICP 初值，看收敛后残差。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]   # research/tools/ → 仓库根
sys.path.insert(0, str(ROOT))

SID = sys.argv[1] if len(sys.argv) > 1 else "20261005_235237"
NPER = 2500
VOX = 0.05


def pick_rec(sid: str) -> Path:
    cand = sorted(p for p in (ROOT / "navmesh_recordings").iterdir()
                  if p.is_dir() and (p / "kf").is_dir() and p.name[:8] == sid[:8])
    return min(cand, key=lambda p: abs(int(p.name[9:]) - int(sid[9:])))


REC = pick_rec(SID)
z = np.load(ROOT / "navmesh_memory" / "wrld_home-7cf435ea" / "sessions" / SID / "poses.npz")
ids = np.asarray(z["ids"]); d = np.asarray(z["dist_m"], float)
T_map = np.asarray(z["T_map"], np.float64); T_dr = np.asarray(z["T_dr"], np.float64)
o = np.argsort(d)
ids, d, T_map, T_dr = ids[o], d[o], T_map[o], T_dr[o]
print(f"会话 {SID} · 录制 {REC.name} · {len(ids)} kf")

rng = np.random.default_rng(3)


def load_pts(i: int) -> np.ndarray:
    with np.load(REC / "kf" / f"{ids[i]:06d}.npz") as kf:
        p = np.asarray(kf["pts"], np.float64)
    p = p[np.isfinite(p).all(1)]
    p = p[(np.abs(p[:, 0]) < 12) & (np.abs(p[:, 1]) < 12) & (p[:, 2] < 20)]
    if len(p) > NPER:
        p = p[rng.choice(len(p), NPER, replace=False)]
    key = np.floor(p / VOX).astype(np.int32)
    _, uniq = np.unique(key, axis=0, return_index=True)
    return p[uniq]


def icp(A: np.ndarray, B: np.ndarray, R0: np.ndarray, t0: np.ndarray, iters: int = 40):
    """点到点 ICP：求 R,t 使 R@A+t ≈ B。返回 (R, t, rmse)。"""
    tree = cKDTree(B)
    R, t = R0.copy(), t0.copy()
    for _ in range(iters):
        P = A @ R.T + t
        dist, idx = tree.query(P, workers=-1)
        keep = dist < np.percentile(dist, 90)          # 剔最差 10%（遮挡/离群）
        Q = B[idx[keep]]
        Pa, Qa = A[keep].mean(0), Q.mean(0)
        H = (A[keep] - Pa).T @ (Q - Qa)
        U, _, Vt = np.linalg.svd(H)
        if np.linalg.det(Vt.T @ U.T) < 0:
            Vt[-1] *= -1
        R = Vt.T @ U.T
        t = Qa - R @ Pa
    P = A @ R.T + t
    return R, t, float(np.sqrt((cKDTree(B).query(P, workers=-1)[0] ** 2).mean()))


# ---------- 挑重访对 ----------
Pm = T_map[:, :2, 3]
yaw = np.arctan2(T_map[:, 2, 0], T_map[:, 0, 0])
tree = cKDTree(Pm)
pr = np.array(tree.query_pairs(1.5, output_type="ndarray"))
far = np.abs(d[pr[:, 0]] - d[pr[:, 1]]) > 30.0
dyaw = np.abs(np.degrees(np.arctan2(np.sin(yaw[pr[:, 0]] - yaw[pr[:, 1]]),
                                    np.cos(yaw[pr[:, 0]] - yaw[pr[:, 1]]))))
print(f"  位置<1.5m 且里程差>30m 的对：{far.sum()}  · 其中朝向差<30°：{(far & (dyaw < 30)).sum()}"
      f"  <60°：{(far & (dyaw < 60)).sum()}")
pr = pr[far & (dyaw < 30)]
if len(pr) > 8:
    pr = pr[rng.choice(len(pr), 8, replace=False)]
print(f"重访对 {len(pr)} 组（**已控朝向 <30°**）· 里程差中位 "
      f"{np.median(np.abs(d[pr[:,0]]-d[pr[:,1]])) if len(pr) else 0:.0f} m\n")

deltas = [0.0, 0.2, 0.5, 1.0, 2.0]
print(f"{'扰动δ(m)':<10}{'ICP后残差rmse(m)':>18}{'平移误差(m)':>14}{'相对DR初值':>12}")
rows = {dd: [] for dd in deltas}
for a, b in pr:
    A, B = load_pts(a), load_pts(b)
    Ta, Tb = T_map[a], T_map[b]
    # 真值：a 的相机系 → b 的相机系。world = R@p + t ⇒
    #   p_b = R_bᵀ(R_a p_a + t_a − t_b) ⇒ R_rel = R_bᵀ R_a, t_rel = R_bᵀ(t_a − t_b)
    Ra_, Rb_ = Ta[:3, :3], Tb[:3, :3]
    R_gt = Rb_.T @ Ra_
    t_gt = Rb_.T @ (Ta[:3, 3] - Tb[:3, 3])
    # DR 初值的相对变换（这是系统现在实际能拿到的初值）
    Ra, Rb = T_dr[a][:3, :3], T_dr[b][:3, :3]
    R_dr = Rb.T @ Ra
    t_dr = Rb.T @ (T_dr[a][:3, 3] - T_dr[b][:3, 3])
    err_dr = float(np.linalg.norm(t_dr - t_gt))
    for dd in deltas:
        if dd == 0.0:
            R0, t0 = R_gt.copy(), t_gt.copy()
        else:
            R0 = R_gt @ (np.eye(3) + 0.0)
            t0 = t_gt + rng.normal(0, dd / 3, 3)      # 各轴 δ/3 ⇒ 期望模长 ≈ δ
        R, t, rmse = icp(A, B, R0, t0)
        rows[dd].append((rmse, float(np.linalg.norm(t - t_gt))))

for dd in deltas:
    r_ = np.array(rows[dd])
    print(f"{dd:<10.1f}{np.median(r_[:,0]):>18.3f}{np.median(r_[:,1]):>14.3f}"
          f"{'（DR 初值误差 %.2f m）' % err_dr if dd == deltas[-1] else '':>12}")
