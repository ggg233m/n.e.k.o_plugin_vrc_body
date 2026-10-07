# -*- coding: utf-8 -*-
"""判定性实验：沿漂移方向做一维扫描，看残差是「尖谷」还是「平谷」。

代价 = trimmed-ICP：只取最近距离的**最优 40%** 求 RMS（避开"一半点根本没对应"的稀释）。
扫描四个方向：修正方向(ICP−DR)、相机视线轴、侧向、竖直；范围 −2…+2 m。
* 尖谷 ⇒ 几何在该方向有强约束 ⇒ ICP 能定出平移；
* 平谷 ⇒ 该方向是弱约束（正对墙面时，沿深度挪动只是让点沿墙面滑）⇒ ICP 定不出，会滑。
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]   # research/tools/ → 仓库根
sys.path.insert(0, str(ROOT / "research" / "tools"))
from dense_pose_refine import Session, icp_translation  # noqa: E402

SID = sys.argv[1] if len(sys.argv) > 1 else "20261005_235237"
S = Session(SID)
rng = np.random.default_rng(9)
pdr, pmp = S.p_dr, S.p_map
SCAN = np.arange(-2.0, 2.01, 0.25)

tree = cKDTree(pdr[:, :2])
pr = np.asarray(tree.query_pairs(1.5, output_type="ndarray"))
pr = pr[np.abs(S.dist[pr[:, 0]] - S.dist[pr[:, 1]]) > 25.0]
if len(pr) > 8:
    pr = pr[rng.choice(len(pr), 8, replace=False)]

print(f"会话 {SID} · 扫描 {len(pr)} 组 · 代价=trimmed RMS(最优40%) · 单位 m\n")
summ = {}

for i, j in pr:
    i, j = int(i), int(j)
    _, A = S.cloud(i, 2500, rng)
    _, B = S.cloud(j, 2500, rng)
    d0 = pdr[j] - pdr[i]
    r = icp_translation(A, B, d0)
    if not np.isfinite(r["rmse"]) or r["keep"] < 0.25:
        continue
    tb = cKDTree(B)

    def cost(shift):
        dd, _ = tb.query(A + r["d"] + shift, workers=-1)
        dd = np.sort(dd)
        k = max(50, int(0.4 * len(dd)))
        return float(np.sqrt((dd[:k] ** 2).mean()))

    view = S.R[i][:, 2]          # 相机视线轴（世界系）
    lat = S.R[i][:, 0]
    corr = r["d"] - d0
    corr = corr / (np.linalg.norm(corr) + 1e-9)
    dirs = {"修正方向": corr, "视线轴": view, "侧向": lat, "竖直": np.array([0.0, 0.0, 1.0])}
    print(f"({i},{j}) ICP残差 {r['rmse']:.3f} 内点 {r['keep']:.0%} "
          f"修正 {np.linalg.norm(r['d']-d0):.2f} m")
    for name, u in dirs.items():
        cs = np.array([cost(u * s) for s in SCAN])
        k0 = int(np.argmin(cs))
        # 谷宽：代价 ≤ 1.2×最小值的区间宽度
        band = SCAN[cs <= 1.2 * cs.min()]
        w = float(band.max() - band.min()) if len(band) else 0.0
        print(f"   {name:<6} 最小 {cs.min():.3f} @ {SCAN[k0]:+.2f} m · "
              f"1.2×谷宽 {w:.2f} m · 剖面 "
              + " ".join(f"{v:.2f}" for v in cs[::4]))
        summ.setdefault(name, []).append((w, cs.min()))
    print()

print("汇总 · 1.2×谷宽中位（**越宽=越约束不住**）")
for k, v in summ.items():
    print(f"   {k:<6} 谷宽中位 {np.median([x[0] for x in v]):.2f} m · "
          f"最小代价中位 {np.median([x[1] for x in v]):.3f} m")
