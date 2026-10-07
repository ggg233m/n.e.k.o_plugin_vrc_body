# -*- coding: utf-8 -*-
"""短跨度对照：几帧之内 DR 一定准 ⇒ 用它验 ICP 有没有尺度/单位问题。

若 ICP 在短跨度上 |d_ICP| ≈ |d_DR| ≈ |d_Tmap| ⇒ ICP 标定正确（单位一致、无尺度错），
那么它在长跨度上和 T_map 差 4.9 m 就是 **T_map 的问题**；
反之若短跨度上 ICP 就与两者差一个固定倍率 ⇒ ICP 自己有尺度/单位错误，长跨度结论不成立。
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
rng = np.random.default_rng(21)
pdr, pmp = S.p_dr, S.p_map


def run(pairs, tag):
    print(f"\n【{tag}】{len(pairs)} 组")
    print(f"  {'i':>5}{'j':>6}{'Δ里程':>7}{'|d_DR|':>8}{'|d_Tmap|':>9}{'|d_ICP|':>8}"
          f"{'|ICP−DR|':>9}{'|ICP−Tmap|':>10}{'倍率':>7}{'残差':>7}{'内点':>6}")
    rows = []
    for i, j in pairs:
        i, j = int(i), int(j)
        _, Bj = S.cloud(i, 2500, rng)
        _, A = S.cloud(j, 2500, rng)
        d0 = pdr[j] - pdr[i]
        r = icp_translation(A, Bj, d0)          # A=j 的云配到 B=i 的云 ⇒ d = t_j − t_i
        dmap = pmp[j] - pmp[i]
        if not np.isfinite(r["rmse"]) or r["keep"] < 0.15:
            continue
        ratio = np.linalg.norm(r["d"]) / (np.linalg.norm(d0) + 1e-9)
        rows.append((np.linalg.norm(r["d"] - d0), np.linalg.norm(r["d"] - dmap), ratio))
        print(f"  {i:>5}{j:>6}{abs(S.dist[j]-S.dist[i]):>7.1f}"
              f"{np.linalg.norm(d0):>8.2f}{np.linalg.norm(dmap):>9.2f}"
              f"{np.linalg.norm(r['d']):>8.2f}"
              f"{np.linalg.norm(r['d']-d0):>9.2f}{np.linalg.norm(r['d']-dmap):>10.2f}"
              f"{ratio:>7.2f}{r['rmse']:>7.3f}{r['keep']:>6.0%}")
    if rows:
        a = np.array(rows)
        print(f"  中位：|ICP−DR| {np.median(a[:,0]):.2f} m · |ICP−Tmap| {np.median(a[:,1]):.2f} m"
              f" · |d_ICP|/|d_DR| 倍率 {np.median(a[:,2]):.2f}")


# 短跨度：相隔 3–8 帧（里程差 2–5 m），DR/T_map 都必然准
short = [(k, k + 5) for k in range(40, 640, 60)]
run(short, "短跨度 Δ≈5 帧（DR 必准 ⇒ 验 ICP 标定）")

# 中跨度
mid = [(k, k + 40) for k in range(40, 600, 60)]
run(mid, "中跨度 Δ≈40 帧")

# 长跨度：DR 位置 <1.5 m（真正的重访候选）
tree = cKDTree(pdr[:, :2])
pr = np.asarray(tree.query_pairs(1.5, output_type="ndarray"))
pr = pr[np.abs(S.dist[pr[:, 0]] - S.dist[pr[:, 1]]) > 25.0][:12]
run(pr, "长跨度重访（DR 位置<1.5 m、里程差>25 m）")
