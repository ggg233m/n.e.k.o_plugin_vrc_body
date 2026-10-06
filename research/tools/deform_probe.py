# -*- coding: utf-8 -*-
"""探查：DR 位姿 vs 权威地图位姿 的差，到底是什么形态。

回答三件事，决定后面实验怎么设计：
1. yaw 是否一致（回环只修平移 ⇒ yaw 应恒等）
2. 位移差 (map - dr) 随里程的走向：是常数、线性、还是弯曲
3. 分段是否必要：全局仿射拟合后的残差随里程是否还有结构
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # research/tools/ → 仓库根
sys.path.insert(0, str(ROOT))

SID = sys.argv[1] if len(sys.argv) > 1 else "20261001_044153"
P = ROOT / "navmesh_memory" / "wrld_home-7cf435ea" / "sessions" / SID / "poses.npz"
z = np.load(P)
ids, T_map, T_dr, dist = z["ids"], z["T_map"], z["T_dr"], z["dist_m"]


def yaw_of(T: np.ndarray) -> np.ndarray:
    return np.arctan2(T[:, 2, 0], T[:, 0, 0])


p_map = T_map[:, :2, 3].astype(float)
p_dr = T_dr[:, :2, 3].astype(float)
y_map, y_dr = yaw_of(T_map.astype(float)), yaw_of(T_dr.astype(float))
d = np.asarray(dist, float)
n = len(ids)

print(f"会话 {SID} · 关键帧 {n} · 里程 {d[0]:.1f}→{d[-1]:.1f} m （总 {d[-1]-d[0]:.1f} m）")
print(f"1) yaw 差：max |Δ| = {np.degrees(np.abs(np.arctan2(np.sin(y_map-y_dr), np.cos(y_map-y_dr)))).max():.4f}°"
      f" · 中位 {np.degrees(np.median(np.abs(np.arctan2(np.sin(y_map-y_dr), np.cos(y_map-y_dr))))):.4f}°"
      f"  ⇒ {'yaw 一致（只修平移）' if np.degrees(np.abs(np.arctan2(np.sin(y_map-y_dr), np.cos(y_map-y_dr)))).max() < 0.01 else 'yaw 也被改了'}")

r = p_map - p_dr                      # 需要的修正量
rn = np.linalg.norm(r, axis=1)
print(f"2) 修正量 |r|：中位 {np.median(rn):.3f} m · p90 {np.percentile(rn,90):.3f} m · max {rn.max():.3f} m")

# 全局仿射：p_map ≈ A p_dr + b
X = np.column_stack([p_dr, np.ones(n)])
sol, *_ = np.linalg.lstsq(X, p_map, rcond=None)
A, b = sol[:2, :2].T, sol[2]          # 每行是 [ax, ay, b]
pred = p_dr @ A.T + b
res = np.linalg.norm(pred - p_map, axis=1)
print(f"3) 全局仿射拟合残差：中位 {np.median(res):.3f} m · p90 {np.percentile(res,90):.3f} m · max {res.max():.3f} m")
# 仿射分解
U, S, Vt = np.linalg.svd(A)
sc = float(np.sqrt(np.linalg.det(A))) if np.linalg.det(A) > 0 else float("nan")
rot = np.degrees(np.arctan2(A[1, 0] - A[0, 1], A[0, 0] + A[1, 1]))
shear = (S[0] - S[1]) / (S[0] + S[1])
print(f"   分解：尺度 {(sc-1)*100:+.2f}% · 旋转 {rot:+.2f}° · 剪切 {shear*100:.2f}% · 奇异值 {S[0]:.4f}/{S[1]:.4f}")

# 残差随里程是否还有结构 ⇒ 分段是否必要
K = 8
edges = np.quantile(d, np.linspace(0, 1, K + 1))
print(f"\n4) 全局仿射残差按里程分 {K} 段（看是否还有结构 ⇒ 分段是否必要）")
print("   段  里程区间(m)       n   残差中位  残差p90   修正量中位")
for i in range(K):
    m = (d >= edges[i]) & (d <= edges[i + 1]) if i == K - 1 else (d >= edges[i]) & (d < edges[i + 1])
    if m.sum() < 2:
        continue
    print(f"   {i:>2}  {edges[i]:6.1f}–{edges[i+1]:6.1f}  {m.sum():5d}"
          f"   {np.median(res[m]):7.3f}   {np.percentile(res[m],90):7.3f}   {np.median(rn[m]):7.3f}")

# 修正量方向是否随里程旋转 ⇒ 是否非刚体
print("\n5) 修正量方向 vs 里程（非刚体 ⇒ 方向系统性旋转）")
for i in range(K):
    m = (d >= edges[i]) & (d <= edges[i + 1]) if i == K - 1 else (d >= edges[i]) & (d < edges[i + 1])
    if m.sum() < 2 or np.median(rn[m]) < 1e-6:
        continue
    v = r[m].mean(0)
    ang = np.degrees(np.arctan2(v[1], v[0]))
    print(f"   段{i:>2} 平均修正 {np.linalg.norm(v):6.3f} m @ {ang:7.1f}°")
