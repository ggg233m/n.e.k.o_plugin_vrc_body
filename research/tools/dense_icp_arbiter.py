# -*- coding: utf-8 -*-
r"""判定：ICP 收敛解与 T_map 相差 1.83 m —— **谁对？**

若直接信 ICP，可能踩经典的"沿弱约束方向滑动"坑：走廊里两面平行墙，沿走廊方向
怎么挪都差不多重合 ⇒ ICP 会滑到"最重合"而非"最正确"。

判据（不靠任何一方当真值）
------------------------
在 ICP 收敛解处**冻结对应关系**，沿某方向平移 s，画残差曲线 f(s)：
* 若沿**偏差方向**挪回 T_map（s = 1.83 m）残差**暴涨** ⇒ 该方向约束强 ⇒ **ICP 明显更优**；
* 若残差**几乎不变** ⇒ 该方向是弱约束/滑动方向 ⇒ **两者都说得通，ICP 不构成证据**。

对照组：沿若干**随机方向**挪同样距离，看残差涨多少。若偏差方向比随机方向还"平"
⇒ 偏差方向正是最弱的方向 ⇒ 滑动。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]   # research/tools/ → 仓库根
sys.path.insert(0, str(ROOT))

SID = sys.argv[1] if len(sys.argv) > 1 else "20261005_235237"
NPER, VOX = 2500, 0.05


def pick_rec(sid):
    cand = sorted(p for p in (ROOT / "navmesh_recordings").iterdir()
                  if p.is_dir() and (p / "kf").is_dir() and p.name[:8] == sid[:8])
    return min(cand, key=lambda p: abs(int(p.name[9:]) - int(sid[9:])))


REC = pick_rec(SID)
z = np.load(ROOT / "navmesh_memory" / "wrld_home-7cf435ea" / "sessions" / SID / "poses.npz")
ids = np.asarray(z["ids"]); d = np.asarray(z["dist_m"], float)
T_map = np.asarray(z["T_map"], np.float64); T_dr = np.asarray(z["T_dr"], np.float64)
o = np.argsort(d)
ids, d, T_map, T_dr = ids[o], d[o], T_map[o], T_dr[o]
rng = np.random.default_rng(11)


def load_pts(i):
    with np.load(REC / "kf" / f"{ids[i]:06d}.npz") as kf:
        p = np.asarray(kf["pts"], np.float64)
    p = p[np.isfinite(p).all(1)]
    p = p[(np.abs(p[:, 0]) < 12) & (np.abs(p[:, 1]) < 12) & (p[:, 2] < 20)]
    if len(p) > NPER:
        p = p[rng.choice(len(p), NPER, replace=False)]
    _, uniq = np.unique(np.floor(p / VOX).astype(np.int32), axis=0, return_index=True)
    return p[uniq]


def icp(A, B, R0, t0, iters=40):
    tree = cKDTree(B)
    R, t = R0.copy(), t0.copy()
    for _ in range(iters):
        P = A @ R.T + t
        dist, idx = tree.query(P, workers=-1)
        keep = dist < np.percentile(dist, 90)
        Q = B[idx[keep]]
        Pa, Qa = A[keep].mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((A[keep] - Pa).T @ (Q - Qa))
        if np.linalg.det(Vt.T @ U.T) < 0:
            Vt[-1] *= -1
        R = Vt.T @ U.T
        t = Qa - R @ Pa
    dist, idx = tree.query(A @ R.T + t, workers=-1)
    keep = dist < np.percentile(dist, 90)
    return R, t, keep, idx[keep], float(np.sqrt((dist[keep] ** 2).mean()))


Pm = T_map[:, :2, 3]
yaw = np.arctan2(T_map[:, 2, 0], T_map[:, 0, 0])
tree = cKDTree(Pm)
pr = np.array(tree.query_pairs(1.5, output_type="ndarray"))
far = np.abs(d[pr[:, 0]] - d[pr[:, 1]]) > 30.0
dyaw = np.abs(np.degrees(np.arctan2(np.sin(yaw[pr[:, 0]] - yaw[pr[:, 1]]),
                                    np.cos(yaw[pr[:, 0]] - yaw[pr[:, 1]]))))
pr = pr[far & (dyaw < 15)]
if len(pr) > 5:
    pr = pr[rng.choice(len(pr), 5, replace=False)]
print(f"会话 {SID} · 重访对 {len(pr)} 组（位置<1.5 m、**朝向差<15°**、里程差>30 m）\n")

SS = [0.0, 0.5, 1.0, 1.5, 2.0]
print(f"{'对':<4}{'|偏差|m':>9}{'f(0)ICP':>10}" + "".join(f"{'s='+format(s,'.1f'):>9}" for s in SS[1:])
      + f"{'随机方向s=1.5':>14}")
for n, (a, b) in enumerate(pr):
    A, B = load_pts(a), load_pts(b)
    Ra_, Rb_ = T_map[a][:3, :3], T_map[b][:3, :3]
    R_gt = Rb_.T @ Ra_
    t_gt = Rb_.T @ (T_map[a][:3, 3] - T_map[b][:3, 3])
    R_icp, t_icp, keep, bidx, rmse0 = icp(A, B, R_gt, t_gt)
    delta = t_icp - t_gt
    dl = float(np.linalg.norm(delta))
    if dl < 1e-6:
        continue
    v = delta / dl
    Qfixed = B[bidx]
    Af = A[keep]

    def f(s, vec):
        P = Af @ R_icp.T + (t_icp + s * vec)
        return float(np.sqrt(((P - Qfixed) ** 2).sum(1).mean()))

    row = f"{n:<4}{dl:>9.2f}{f(0.0, v):>10.3f}"
    for s in SS[1:]:
        row += f"{f(s, v):>9.3f}"
    # 对照：随机方向（垂直于偏差）挪 1.5 m
    vr = rng.normal(size=3); vr -= vr @ v * v; vr /= np.linalg.norm(vr)
    row += f"{f(1.5, vr):>14.3f}"
    print(row)

print("\n读法：从 s=0（ICP 解）沿**偏差方向**挪向 T_map，残差涨得越多 ⇒ 该方向约束越强 ⇒ ICP 越可信；")
print("      若几乎不涨、且比随机方向还平 ⇒ 偏差方向正是滑动方向 ⇒ ICP 不构成证据。")
