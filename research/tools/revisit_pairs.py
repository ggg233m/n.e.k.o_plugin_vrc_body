# -*- coding: utf-8 -*-
r"""回访错位：**强制同一批回访对**，跨多个位姿表对比。

为什么需要它（2026-10-27 实测）：现有的 revisit_shift.py 让**每个位姿源各自挑对**，
结果是 DR 有 36 对、回环只有 20 对、vo_g6 只有 14 对 —— 三个数不是同一个样本，
把它们并排写进一张表就是错的（我曾据此得出「回环校正 54 cm vs 基线 177 cm」的错误结论，
实际两者帧集都不同）。本工具先按**参考位姿**挑回访对，再要求所有位姿源都有这些帧，
最后只在**交集**上算，保证每行是同一个样本。

口径：严格墙面（离地 0.3–1.6 世界米、法向倾角 ≤3.4°，与 FACTS §六 的教训一致）。
默认参考位姿 = final_poses_full（缺则用 kf 自带的 T_dr）。

用法：
    .venv/Scripts/python.exe research/tools/revisit_pairs.py 20261026_184318
    .venv/Scripts/python.exe research/tools/revisit_pairs.py 20261026_184318 \
        --poses final_poses full joint_poses vo_g6

两列必须都读：「相邻帧失配」与「回访错位」是**两个自由度**，一个变好另一个可以变坏。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT));
sys.path.insert(0, str(ROOT / "research" / "tools"))
import smear_eval as SE  # noqa: E402

S_WORLD = 0.755      # 追踪米 → 世界米；FACTS §二
CAM_H = 1.75         # 地图单位；严格口径常数（.tmp/smear_strict.py:8）


def strict_wall(F: dict) -> dict:
    """只保留离地 0.3–1.6 世界米的点，拟严格竖直面（|nz|<0.06 ⇒ 倾角 ≤3.4°）。

    ⚠️ 不要用 SE.fit_vertical_plane 的默认 max_vz=0.35（允许 ~20° 倾斜）——
    实测它常拟合到**地面**，法向倾角中位 14.4°，结论会整个作废（FACTS §六）。
    """
    rng = np.random.default_rng(0)
    for k, r in F.items():
        P = r["pts"]
        z = P[:, 2] + CAM_H
        r["w"] = P[(z > 0.3 / S_WORLD) & (z < 1.6 / S_WORLD)]
        r["pl"] = SE.fit_vertical_plane(r["w"], rng, max_vz=0.06, min_in=600)
    return F


def wall_shift(PA: np.ndarray, PB: np.ndarray, n: np.ndarray, p0: np.ndarray):
    """两片严格墙面点集沿法向的**带符号**中位差（同 smear_strict.smed 口径）。"""

    def agg(P):
        e1 = np.cross(n, [0.0, 0.0, 1.0])
        n1 = np.linalg.norm(e1)
        if n1 < 1e-9:
            return None
        e1 = e1 / n1
        E = np.column_stack([e1, np.cross(n, e1)])
        X = P - p0
        kk = np.floor(X @ E / SE.CELL).astype(np.int64)
        key = kk[:, 0] * 1_000_003 + kk[:, 1]
        u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        h = X @ n
        o = np.lexsort((h, inv))
        st = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        hs = h[o]
        return u, cnt, 0.5 * (hs[st + (cnt - 1) // 2] + hs[st + cnt // 2])

    ra, rb = agg(PA), agg(PB)
    if ra is None or rb is None:
        return None
    ua, ca, ma = ra
    ub, cb, mb = rb
    _, ia, ib = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    ok = (ca[ia] >= SE.MINPTS) & (cb[ib] >= SE.MINPTS) & (np.abs(ma[ia]) < 0.1)
    if int(ok.sum()) < 5:
        return None
    return float(np.median(mb[ib][ok] - ma[ia][ok]))


def load_pose(rec: Path, name: str) -> dict | None:
    p = rec / f"{name}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {int(k): np.asarray(T, float) for k, T in zip(z["ids"], z["T"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rec", help="录制目录名（navmesh_recordings/ 下）")
    ap.add_argument("--poses", nargs="+",
                    default=["final_poses", "joint_poses", "replay_poses", "vo_g6", "vo3_poses"],
                    help="要对比的 <rec>/<name>.npz（缺的自动跳过）")
    ap.add_argument("--ref", default="final_poses_full",
                    help="挑回访对用的参考位姿表（默认 final_poses_full）")
    ap.add_argument("--dt", type=float, default=10.0, help="回访对的最小时间差 s")
    ap.add_argument("--near", type=float, default=2.0, help="回访对的空间距离上限 m")
    ap.add_argument("--max-pairs", type=int, default=120)
    a = ap.parse_args()

    rec = SE.REC_ROOT / a.rec
    if not rec.is_dir():
        raise SystemExit(f"[fail] 没有这个录制：{rec}")
    F = strict_wall(SE.load_cached(rec))
    ids = sorted(F)
    n_wall = sum(1 for r in F.values() if r["pl"] is not None)
    print(f"{a.rec}：严格墙面 {n_wall}/{len(ids)} 帧")
    if n_wall < 20:
        print("[warn] 严格墙面太少，结论不稳")

    srcs = {}
    ref = load_pose(rec, a.ref)
    if ref is None:
        print(f"[warn] 没有 {a.ref}.npz，参考位姿退回 kf 自带的 T_dr")
        ref = {k: F[k]["T"] for k in ids}
    srcs["DR(参考)"] = ref
    for nm in a.poses:
        tp = load_pose(rec, nm)
        if tp is None:
            print(f"[skip] 没有 {nm}.npz")
            continue
        srcs[nm] = tp

    tt = {k: F[k]["t"] for k in ids}
    pairs = []
    for i, ka in enumerate(ids):
        if F[ka]["pl"] is None:
            continue
        for kb in ids[i + 1:]:
            if len(pairs) >= a.max_pairs:
                break
            if F[kb]["pl"] is None:
                continue
            if tt[kb] - tt[ka] < a.dt:
                continue
            if ka not in ref or kb not in ref:
                continue
            if float(np.hypot(*(ref[kb][:2, 3] - ref[ka][:2, 3]))) < a.near:
                pairs.append((ka, kb))
        if len(pairs) >= a.max_pairs:
            break
    print(f"候选回访对 {len(pairs)}（参考={a.ref}，dt≥{a.dt}s，距离≤{a.near}m）")

    okpairs = [(ka, kb) for ka, kb in pairs
               if all(ka in s and kb in s for s in srcs.values())]
    print(f"所有位姿源都覆盖的对 {len(okpairs)}（**同一批样本**，行与行可比）")
    if not okpairs:
        print("[fail] 没有共同覆盖的回访对；减少 --poses 或换 --ref")
        return 1

    print()
    print(f"{'位姿源':14s} {'有效对':>6s} {'|偏移|中位':>11s} {'p75':>9s}")
    for nm, TP in srcs.items():
        vals = []
        for ka, kb in okpairs:
            Ta, Tb = TP[ka], TP[kb]
            nb, p0b, _ = F[ka]["pl"]
            n = Ta[:3, :3] @ nb
            p0 = Ta[:3, :3] @ p0b + Ta[:3, 3]
            if (p0 - Ta[:3, 3]) @ n < 0:
                n = -n
            PA = F[ka]["w"] @ Ta[:3, :3].T + Ta[:3, 3]
            PB = F[kb]["w"] @ Tb[:3, :3].T + Tb[:3, 3]
            m = wall_shift(PA, PB, n, p0)
            if m is not None:
                vals.append(abs(m) * S_WORLD)
        if vals:
            v = np.asarray(vals)
            warn = "  ⚠️ 样本<15，不稳" if len(v) < 15 else ""
            print(f"{nm:14s} {len(v):6d} {np.median(v)*100:9.1f}cm "
                  f"{np.percentile(v,75)*100:8.1f}{warn}")
        else:
            print(f"{nm:14s} {0:6d}   [无有效对]")
    print()
    print("判读：所有行都是同一批对，可直接横比。")
    print("⚠️ 只有这一列**不能**判位姿好坏 —— 必须同时看相邻帧失配（smear_eval）与面厚。")
    if len(okpairs) < 40:
        print(f"⚠️ 共同对只有 {len(okpairs)} 个。--poses 每多一个源，交集就再小一圈；")
        print("   想要大样本就**分两批**跑，或减少 --poses。**不要**拿小样本的排名当结论。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
