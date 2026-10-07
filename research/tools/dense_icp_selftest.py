# -*- coding: utf-8 -*-
r"""ICP 实现自检：**合成已知平移**，看能不能恢复到厘米级。

用途：当真实数据上的配准结果可疑时，先跑它排除"是我写错了"。
同一片云随机切成两半、一半平移已知量 t，ICP 从 d0=0 起步应恢复到 ~0.05 m。
若这里就失败 ⇒ 实现有 bug；若这里通过而真实数据失败 ⇒ **场景/数据问题**（部分重叠、
长直墙导致的沿墙退化），不是代码问题。

2026-10-06 实测：0.058 / 0.060 / 0.064 / 0.058 m，内点 99% —— 通过。
而同一实现在真实相邻关键帧上（DR 当作真值）误差 0.56–1.88 m ⇒ 出局的是路线不是代码。

用法::

    python research/tools/dense_icp_selftest.py [sid]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]   # research/tools/ → 仓库根
sys.path.insert(0, str(ROOT / "research" / "tools"))
from dense_pose_refine import Session, icp_translation  # noqa: E402

SID = sys.argv[1] if len(sys.argv) > 1 else "20261005_235237"
S = Session(SID)
rng = np.random.default_rng(0)
_, Q = S.cloud(300, 20000, rng)          # 注意：nper=0 会被当成"抽 0 个点"，别传 0

print(f"会话 {SID} · 自检云 {len(Q)} 点 · 切成两半、一半平移已知量 t，ICP 从 d0=0 起步")
print(f"{'t_true':>20}{'ICP 解':>22}{'误差 m':>9}{'残差':>8}{'内点':>7}")
worst = 0.0
for t in ([0.4, 0.1, 0.0], [-1.2, 0.5, 0.0], [2.0, -1.5, 0.0], [0.0, 0.0, 0.5]):
    t = np.asarray(t, np.float64)
    idx = rng.permutation(len(Q))
    A = Q[idx[: len(Q) // 2]] + t
    B = Q[idx[len(Q) // 2:]]
    r = icp_translation(A, B, np.zeros(3))
    err = float(np.linalg.norm(r["d"] + t))
    worst = max(worst, err)
    print(f"{str(np.round(t, 2)):>20}{str(np.round(r['d'], 2)):>22}"
          f"{err:>9.3f}{r['rmse']:>8.3f}{r['keep']:>7.0%}")
print(f"\n最大误差 {worst:.3f} m —— " + ("✅ 实现正常（真实数据上的失败归咎于场景/数据）"
                                        if worst < 0.15 else "❌ 实现有问题，先修代码"))
sys.exit(0 if worst < 0.15 else 1)
