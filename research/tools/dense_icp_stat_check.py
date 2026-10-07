# -*- coding: utf-8 -*-
"""对质：昨天「偏差方向不是弱约束」的结论，是不是被**截尾中位**这个统计量骗出来的？

同一批重访对、同一对云，同时算两种残差统计：
  A. 截尾中位（昨天用的）：median(dist[dist < 1.0])
  B. 截断均值（今天用的）：mean(min(dist, 1.0))
沿三个方向画剖面：修正方向(ICP−DR)、指向 T_map 的方向、随机方向。
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
rng = np.random.default_rng(5)
pmp, pdr = S.p_map, S.p_dr

tree = cKDTree(pdr[:, :2])
pr = np.asarray(tree.query_pairs(1.5, output_type="ndarray"))
pr = pr[np.abs(S.dist[pr[:, 0]] - S.dist[pr[:, 1]]) > 25.0]
if len(pr) > 10:
    pr = pr[rng.choice(len(pr), 10, replace=False)]

print(f"会话 {SID} · 重访对 {len(pr)} 组")
print("列：方向 → 剖面(0 / 0.25 / 0.5 / 1.0 m)；统计 A=截尾中位(昨) B=截断均值(今)\n")

for i, j in pr:
    i, j = int(i), int(j)
    _, A = S.cloud(i, 2500, rng)
    _, B = S.cloud(j, 2500, rng)
    d0 = pdr[j] - pdr[i]
    r = icp_translation(A, B, d0)
    if not np.isfinite(r["rmse"]) or r["keep"] < 0.2:
        continue
    tb = cKDTree(B)

    def prof(unit, s):
        dd, _ = tb.query(A + r["d"] + unit * s, workers=-1)
        a = float(np.median(dd[dd < 1.0])) if (dd < 1.0).any() else 1.0
        b = float(np.mean(np.minimum(dd, 1.0)))
        return a, b

    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None

    dirs = {}
    u = unit(r["d"] - d0)
    if u is not None:
        dirs["修正(ICP−DR)"] = u
    u = unit((pmp[j] - pmp[i]) - r["d"])
    if u is not None:
        dirs["指向T_map"] = u
    ur = rng.normal(size=3)
    dirs["随机"] = ur / np.linalg.norm(ur)

    print(f"({i},{j}) ICP残差 {r['rmse']:.3f} 内点 {r['keep']:.0%} "
          f"修正 {np.linalg.norm(r['d']-d0):.2f} m")
    for name, uu in dirs.items():
        pa = [prof(uu, s)[0] for s in (0.0, 0.25, 0.5, 1.0)]
        pb = [prof(uu, s)[1] for s in (0.0, 0.25, 0.5, 1.0)]
        ga = pa[-1] / pa[0] if pa[0] > 1e-6 else 0
        gb = pb[-1] / pb[0] if pb[0] > 1e-6 else 0
        print(f"   {name:<12} A " + " ".join(f"{v:.3f}" for v in pa) + f"  (×{ga:.2f})"
              + " | B " + " ".join(f"{v:.3f}" for v in pb) + f"  (×{gb:.2f})")
    print()
