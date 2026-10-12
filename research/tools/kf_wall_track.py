# -*- coding: utf-8 -*-
r"""逐帧（因果）墙面跟踪：新关键帧的墙带点对前几帧已校正的局部墙图做 2D 点到线 ICP，
只解平移（旋转用 HMD），DR 预测（×osc 系数）当先验一起解 —— 退化方向（平行走廊）自动由先验兜住。

只用过去的帧 ⇒ 可直接搬进在线；产出 <rec>/<out>.npz（ids, T），可喂 smear_strict / grid_pose_ab。
⚠️ 输出名别以 dr 开头（grid_pose_ab.poses_for 会当原始 DR）。

    .venv/Scripts/python.exe research/tools/kf_wall_track.py 20261010_033237 --out wt_poses.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
REC_ROOT = ROOT / "navmesh_recordings"
S = 0.755          # 地图单位 → 世界米
CAMH = 1.75        # 相机离地（地图单位，5 场一致）


def wall2d(P: np.ndarray, R: np.ndarray, *, hlo=0.3, hhi=1.8, rlo=0.5, rhi=4.0, cell=0.04) -> np.ndarray:
    """kf 点（相机系）→ 旋转到世界朝向后的墙带 2D 点（未加平移），4 cm 栅格去重。参数为世界米。"""
    if len(P) == 0:
        return np.zeros((0, 2))
    W = P @ R.T
    h = W[:, 2] + CAMH
    r = np.hypot(P[:, 0], P[:, 1])
    W = W[(h > hlo / S) & (h < hhi / S) & (r > rlo / S) & (r < rhi / S)]
    q = np.unique(np.floor(W[:, :2] / (cell / S)).astype(np.int64), axis=0)
    return (q + 0.5) * (cell / S)


def normals(X: np.ndarray, tree: cKDTree, rad: float = 0.15 / S):
    N = np.zeros_like(X)
    ok = np.zeros(len(X), bool)
    for i, nb in enumerate(tree.query_ball_point(X, rad)):
        if len(nb) < 5:
            continue
        w, v = np.linalg.eigh(np.cov(X[nb].T))
        if w[0] < 0.15 * w[1]:
            N[i] = v[:, 0]
            ok[i] = True
    return N, ok


def register(B: np.ndarray, A: np.ndarray, NA: np.ndarray, tree: cKDTree, t0: np.ndarray, t_pred: np.ndarray,
             sig_pr: float, *, sig_icp=0.05 / S, gate=0.3 / S, iters=15):
    """argmin Σ huber(n·(B+t−A))/σ_icp² + |t−t_pred|²/σ_pr²。返回 (t, 内点数, 视觉信息矩阵)。"""
    t = t0.copy()
    H = np.zeros((2, 2))
    m_cnt = 0
    for _ in range(iters):
        X = B + t
        d, j = tree.query(X, distance_upper_bound=gate)
        m = np.isfinite(d)
        m[m] = NA[j[m]].any(1)
        m_cnt = int(m.sum())
        Hp = np.eye(2) / sig_pr ** 2
        gp = (t - t_pred) / sig_pr ** 2
        if m_cnt < 30:
            H = np.zeros((2, 2))
            dx = -np.linalg.solve(Hp, gp)
        else:
            n = NA[j[m]]
            r = np.einsum("ij,ij->i", X[m] - A[j[m]], n)
            w = 1.0 / np.maximum(np.abs(r) / sig_icp, 1.0) / sig_icp ** 2
            # 每个点的权重按点数归一到"等效 ~200 个独立观测"，免得几千个相关点把先验压没
            w *= min(1.0, 200.0 / m_cnt)
            H = (n * w[:, None]).T @ n
            g = (n * w[:, None]).T @ r
            dx = -np.linalg.solve(H + Hp, g + gp)
        t = t + dx
        if np.hypot(*dx) < 1e-3:
            break
    return t, m_cnt, H


def load(rec: Path):
    ev = [json.loads(x) for x in (rec / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    kt = {int(e["k"]): float(e["t"]) for e in ev if e.get("kind") == "kf"}
    F = {}
    for p in sorted((rec / "kf").glob("*.npz"), key=lambda q: int(q.stem)):
        k = int(p.stem)
        if k not in kt:
            continue
        d = np.load(p)
        F[k] = {"T": np.asarray(d["T_dr"], float), "pts": d["pts"].astype(np.float64), "t": kt[k]}
    return F


def track(F: dict, *, k_osc=0.85, nwin=6, sig0=0.03, sig_rel=0.15, max_innov=0.5, win_dt=6.0, log=None):
    """返回 {k: T}。先验 = 上一帧校正位姿 + DR 增量×k_osc；σ_pr = sig0 + sig_rel·|增量|（世界米）。"""
    ids = sorted(F)
    out, walls, stats = {}, {}, []
    for i, k in enumerate(ids):
        Tdr = F[k]["T"]
        R = Tdr[:3, :3]
        walls[k] = wall2d(F[k]["pts"], R)
        if i == 0:
            out[k] = Tdr.copy()
            continue
        kp = ids[i - 1]
        step = (Tdr[:2, 3] - F[kp]["T"][:2, 3]) * k_osc
        t_pred = out[kp][:2, 3] + step
        sig_pr = (sig0 + sig_rel * np.hypot(*step) * S) / S
        win = [q for q in ids[max(0, i - nwin):i] if F[k]["t"] - F[q]["t"] < win_dt]
        A = np.concatenate([walls[q] + out[q][:2, 3] for q in win]) if win else np.zeros((0, 2))
        t, nin, Hv = t_pred, 0, np.zeros((2, 2))
        if len(A) >= 200 and len(walls[k]) >= 100:
            A = np.unique(np.floor(A / (0.04 / S)).astype(np.int64), axis=0) * (0.04 / S) + 0.02 / S
            tree = cKDTree(A)
            NA, ok = normals(A, tree)
            NA[~ok] = 0.0
            t, nin, Hv = register(walls[k], A, NA, tree, t_pred, t_pred, sig_pr)
            if np.hypot(*(t - t_pred)) * S > max_innov:      # 跳太远 ⇒ 不信视觉
                t, nin, Hv = t_pred, -1, np.zeros((2, 2))
        T = Tdr.copy()
        T[:2, 3] = t
        out[k] = T
        stats.append((np.hypot(*(t - t_pred)) * S, nin, np.sqrt(max(np.linalg.eigvalsh(Hv)[0], 0)) * sig_pr))
    st = np.array(stats)
    if log is not None:
        log(f"  innov 中位 {np.median(st[:,0])*100:.1f} cm p90 {np.percentile(st[:,0],90)*100:.1f}；"
            f"视觉帧 {np.mean(st[:,1]>=30)*100:.0f}%；拒绝 {np.sum(st[:,1]<0)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="+")
    ap.add_argument("--k-osc", type=float, default=0.85)
    ap.add_argument("--nwin", type=int, default=6)
    ap.add_argument("--sig-rel", type=float, default=0.15)
    ap.add_argument("--out", default="wt_poses.npz")
    a = ap.parse_args()
    for name in a.recs:
        rec = REC_ROOT / name
        t0 = time.perf_counter()
        F = load(rec)
        P = track(F, k_osc=a.k_osc, nwin=a.nwin, sig_rel=a.sig_rel, log=print)
        ids = sorted(P)
        np.savez(rec / a.out, ids=np.array(ids), T=np.stack([P[k] for k in ids]))
        dt = time.perf_counter() - t0
        print(f"{name}: {len(ids)} 帧 {dt:.1f} s（{dt/len(ids)*1000:.0f} ms/帧）→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
