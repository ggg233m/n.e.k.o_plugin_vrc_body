# -*- coding: utf-8 -*-
"""最后一搏：短跨度对照下，ICP 错 1.88 m —— 是**远场噪声+抽样太稀**还是**场景退化**？

三个变体 × 短跨度对照（DR=真值）：
  A. 基线        ：2500 点、深度不限
  B. 近场        ：8000 点、**只留深度<3 m**（1 px 视差在 2 m 处已 0.16 m，远场是噪声）
  C. 近场+全密度 ：不限抽样，全部深度<3 m 的点（~1.5–2 万）
若 B/C 的 |ICP−DR| 掉到 0.1–0.2 m ⇒ 只是抽样/远场问题，路线可救；
若仍 ~1 m ⇒ **场景退化（长直墙：沿墙平移不改残差）**，路线出局。
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
pdr = S.p_dr


def cloud(i, nper, dmax):
    with np.load(S.rec / "kf" / f"{S.ids[i]:06d}.npz") as kf:
        p = np.asarray(kf["pts"], np.float64)
    p = p[np.isfinite(p).all(1)]
    p = p[(np.abs(p[:, 0]) < 12) & (np.abs(p[:, 1]) < 12)
          & (np.abs(p[:, 2]) < dmax) & (p[:, 2] > 0.3)]
    n0 = len(p)
    if nper and len(p) > nper:
        p = p[np.random.default_rng(i).choice(len(p), nper, replace=False)]
    return p @ S.R[i].T, n0


def variant(name, nper, dmax, pairs):
    errs, ratios = [], []
    for i, j in pairs:
        i, j = int(i), int(j)
        Bj, n0 = cloud(i, nper, dmax)
        A, _ = cloud(j, nper, dmax)
        if len(A) < 300 or n0 < 300:
            continue
        d0 = pdr[j] - pdr[i]
        r = icp_translation(A, Bj, d0)
        if not np.isfinite(r["rmse"]) or r["keep"] < 0.15:
            continue
        errs.append(np.linalg.norm(r["d"] - d0))
        ratios.append(np.linalg.norm(r["d"]) / (np.linalg.norm(d0) + 1e-9))
    if errs:
        e = np.array(errs)
        print(f"  {name:<14} n={len(e):>2} · |ICP−DR| 中位 {np.median(e):.2f} m · "
              f"p90 {np.percentile(e,90):.2f} · 倍率中位 {np.median(ratios):.2f} · "
              f"<0.3 m 占 {np.mean(e < 0.3):.0%}")


short = [(k, k + 5) for k in range(40, 640, 30)]
print(f"会话 {SID} · 短跨度对照 {len(short)} 组（相隔 5 帧，DR 误差 <10 cm ⇒ 当真值）")
variant("A 基线 2500", 2500, 12.0, short)
variant("B 近场 8000", 8000, 3.0, short)
variant("C 近场全密度", 0, 3.0, short)
variant("D 全密度全深度", 0, 12.0, short)
