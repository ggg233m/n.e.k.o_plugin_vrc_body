# -*- coding: utf-8 -*-
"""DR vs T_map（回环产物）：谁更几何自洽？——**对称**裁判，绕开选择偏差。

偏差问题：只按 DR 挑"重访对"会天然偏向 DR。这里两组都做：
  A 组：DR 位置<1.5 m 的对      （偏 DR）
  B 组：T_map 位置<1.5 m 的对   （偏 T_map）
交叉看谁赢：
  * DR 在两组都赢 ⇒ 回环在帮倒忙；
  * 各自在自己的组赢 ⇒ 选择偏差主导，本裁判判不了（但分歧幅度本身仍是信息）。

裁判 = trimmed RMS（最近距离的最优 40% 求 RMS），不依赖任何 ICP 收敛。
"""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]   # research/tools/ → 仓库根
sys.path.insert(0, str(ROOT / "research" / "tools"))
from dense_pose_refine import Session  # noqa: E402

SID = sys.argv[1] if len(sys.argv) > 1 else "20261005_235237"
S = Session(SID)
rng = np.random.default_rng(31)
pdr, pmp = S.p_dr, S.p_map
NP, MAXP = 3000, 20

dR = np.abs(np.diff(np.unwrap([
    np.arctan2(S.R[i][1, 0], S.R[i][0, 0]) for i in range(S.n)])))
yaw = np.concatenate([[0.0], np.cumsum(np.degrees(dR))])


def judge(i, j, t):
    _, Ai = S.cloud(i, NP, rng)
    _, Aj = S.cloud(j, NP, rng)
    dd, _ = cKDTree(Aj + t[j]).query(Ai + t[i], workers=-1)
    dd = np.sort(dd)
    k = max(50, int(0.4 * len(dd)))
    return float(np.sqrt((dd[:k] ** 2).mean()))


def pairs_by(t, tag):
    tree = cKDTree(t[:, :2])
    pr = np.asarray(tree.query_pairs(1.5, output_type="ndarray"))
    dyaw = np.abs(yaw[pr[:, 0]] - yaw[pr[:, 1]]) % 360
    dyaw = np.minimum(dyaw, 360 - dyaw)
    pr = pr[(np.abs(S.dist[pr[:, 0]] - S.dist[pr[:, 1]]) > 25.0) & (dyaw < 15.0)]
    if len(pr) > MAXP:
        pr = pr[rng.choice(len(pr), MAXP, replace=False)]
    print(f"\n【{tag}】{len(pr)} 组")
    print(f"  {'i':>5}{'j':>6}{'DR':>8}{'T_map':>8}{'相差':>8}")
    dr_w, mp_w = 0, 0
    vd, vm = [], []
    for i, j in pr:
        i, j = int(i), int(j)
        a, b = judge(i, j, pdr), judge(i, j, pmp)
        vd.append(a); vm.append(b)
        if a < b * 0.8:
            dr_w += 1
        elif b < a * 0.8:
            mp_w += 1
        print(f"  {i:>5}{j:>6}{a:>8.3f}{b:>8.3f}{abs(a-b):>8.3f}")
    print(f"  中位：DR {np.median(vd):.3f} · T_map {np.median(vm):.3f} · "
          f"DR 明显赢 {dr_w}/{len(pr)} · T_map 明显赢 {mp_w}/{len(pr)}")
    return np.median(vd), np.median(vm)


print(f"会话 {SID} · 裁判=trimmed RMS(最优40%) · 越低越几何自洽")
pairs_by(pdr, "A 组：按 DR 位置<1.5 m 挑（偏 DR）")
pairs_by(pmp, "B 组：按 T_map 位置<1.5 m 挑（偏 T_map）")
