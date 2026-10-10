# -*- coding: utf-8 -*-
r"""关键帧间**点到面**平移配准（退化感知）→ 平移位姿图 → 修正航位推算位姿（原型）。

与已证伪的稠密 ICP（`Docs/archive/稠密配准约束源-证伪（2026-10-06）.md`）的区别：
那次失败是"长直墙 ⇒ 沿墙平移不改残差 ⇒ 残差漂亮但平移错"。本工具：
 * 只用**点到面**残差 r = n·(p_b + δ − q_a)，信息矩阵 H = Σ w·n nᵀ。
   沿墙方向 n 无分量 ⇒ H 在该方向**特征值≈0** ⇒ 位姿图在该方向完全听 DR。
   ⇒ 退化方向**不是被估错，而是根本不估**。
 * 只做相邻 span 帧（DR 帧间 ≲ 0.5 m，初值在收敛域内），只解水平 xy（DR 里 z 恒 0）。
 * 修正量 > max_corr 的约束丢弃。

    .venv/Scripts/python.exe research/tools/kf_plane_align.py 20261001_044153
    # --holdout：每帧把"最大竖直面"附近的点剔掉再配准（与 .tmp/pc_smear_pair.py 的度量面同一选法），
    #             防止"修的就是被量的那面墙"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
REC_ROOT = ROOT / "navmesh_recordings"


def voxel(P, s):
    k = np.floor(P / s).astype(np.int64)
    _, i = np.unique(k, axis=0, return_index=True)
    return P[i]


def normals(P, tree, k=12):
    _, nb = tree.query(P, k=k)
    Q = P[nb] - P[nb].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q) / k
    w, V = np.linalg.eigh(C)
    n = V[:, :, 0]
    flat = w[:, 0] / np.maximum(w.sum(1), 1e-12)      # 曲率：越小越平
    return n, flat


def biggest_vertical_plane(P, rng, thr=0.05, iters=400):
    """与度量脚本同口径的粗选：最大竖直面（只用于 --holdout 剔除）。"""
    if len(P) < 200:
        return None
    Q = P[rng.choice(len(P), size=min(4000, len(P)), replace=False)]
    best, bm = 0, None
    for _ in range(iters):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-9 or abs(n[2] / ln) > 0.35:
            continue
        n /= ln
        cnt = int((np.abs((Q - a) @ n) < thr).sum())
        if cnt > best:
            best, bm = cnt, (n, a)
    return bm


def load(rec, rng, *, vox, rmax, holdout):
    ev = [json.loads(x) for x in (rec / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    kt = {int(e["k"]): float(e["t"]) for e in ev if e.get("kind") == "kf"}
    F = {}
    for p in sorted((rec / "kf").glob("*.npz"), key=lambda q: int(q.stem)):
        k = int(p.stem)
        if k not in kt:
            continue
        d = np.load(p)
        P = d["pts"].astype(np.float64)
        P = P[(P[:, 0] > 0.3) & (np.hypot(P[:, 0], P[:, 1]) < rmax)]
        if holdout:
            bm = biggest_vertical_plane(P[(P[:, 0] > 0.6) & (P[:, 0] < 5.0)], rng)
            if bm is not None:
                n, a = bm
                P = P[np.abs((P - a) @ n) > 0.15]
        P = voxel(P, vox)
        T = np.asarray(d["T_dr"], float)
        W = P @ T[:3, :3].T + T[:3, 3]
        F[k] = {"T": T, "W": W, "t": kt[k]}
    return F


def align(a, b, *, iters=8, sig=0.03, max_corr=0.6):
    """b 相对 a 的水平平移修正 δ（世界系 xy）：最小化 Σ ρ(n_a·(W_b + δ − q_a))。返回 (δ, H) 或 None。"""
    if len(a["W"]) < 300 or len(b["W"]) < 300:
        return None
    if "tree" not in a:
        a["tree"] = cKDTree(a["W"])
        a["n"], a["flat"] = normals(a["W"], a["tree"])
    delta = np.zeros(2)
    for it, rad in enumerate(np.linspace(0.5, 0.12, iters)):
        Pb = b["W"].copy()
        Pb[:, :2] += delta
        dist, j = a["tree"].query(Pb, distance_upper_bound=rad)
        ok = np.isfinite(dist)
        ok[ok] &= a["flat"][j[ok]] < 0.05
        if int(ok.sum()) < 200:
            return None
        n = a["n"][j[ok]]
        r = np.einsum("ij,ij->i", n, Pb[ok] - a["W"][j[ok]])
        nxy = n[:, :2]
        w = 1.0 / np.maximum(1.0, np.abs(r) / sig)          # Huber
        H = (nxy * w[:, None]).T @ nxy
        g = (nxy * w[:, None]).T @ r
        ev, V = np.linalg.eigh(H)
        # 退化方向（特征值 < 主方向的 5%）不更新
        step = np.zeros(2)
        for e, v in zip(ev, V.T):
            if e > 0.05 * ev.max():
                step -= v * (v @ g) / e
        delta += step
        if np.linalg.norm(delta) > max_corr:
            return None
    # 有效样本数：邻近体素强相关，按 1/10 计
    Hn = H / sig ** 2 / 10.0
    return delta, Hn, int(ok.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--span", type=int, default=4)
    ap.add_argument("--vox", type=float, default=0.06)
    ap.add_argument("--rmax", type=float, default=3.0, help="水平半径上限（追踪米）；FACTS：3 cm 可信半径 ≈2.5 m")
    ap.add_argument("--dr-sigma", type=float, default=0.3)
    ap.add_argument("--max-dt", type=float, default=3.0)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--out", default="pa_poses.npz")
    a = ap.parse_args()
    rec = REC_ROOT / a.rec
    rng = np.random.default_rng(0)
    F = load(rec, rng, vox=a.vox, rmax=a.rmax, holdout=a.holdout)
    ids = sorted(F)
    n = len(ids)
    cons, mags, ratios, n_try = [], [], [], 0
    for i, k in enumerate(ids):
        for s in range(1, a.span + 1):
            if i + s >= n or F[ids[i + s]]["t"] - F[k]["t"] > a.max_dt:
                break
            n_try += 1
            r = align(F[k], F[ids[i + s]])
            if r is None:
                continue
            d, H, _ = r
            ev = np.linalg.eigvalsh(H)
            ratios.append(ev[0] / max(ev[1], 1e-12))
            d_dr = F[ids[i + s]]["T"][:2, 3] - F[k]["T"][:2, 3]
            cons.append((i, i + s, d_dr + d, H))
            mags.append(float(np.linalg.norm(d)))
        F[k].pop("tree", None)
    mags = np.array(mags)
    print(f"[{a.rec}{' holdout' if a.holdout else ''}] 关键帧 {n}；尝试 {n_try}，接受 {len(cons)}（{100*len(cons)/max(n_try,1):.0f}%）")
    if len(mags):
        print(f"  帧间修正 |δ|：med {np.median(mags)*100:.1f} cm  p90 {np.percentile(mags,90)*100:.1f} cm；"
              f"退化(λmin/λmax<0.05) 占 {100*np.mean(np.array(ratios)<0.05):.0f}%")
    rows = (n - 1) * 2 + len(cons) * 2 + 2
    A = lil_matrix((rows, 2 * n))
    b = np.zeros(rows)
    r = 0
    w = 1.0 / a.dr_sigma
    for i in range(n - 1):
        d = F[ids[i + 1]]["T"][:2, 3] - F[ids[i]]["T"][:2, 3]
        for c in range(2):
            A[r, 2 * (i + 1) + c] = w
            A[r, 2 * i + c] = -w
            b[r] = w * d[c]
            r += 1
    for i, j, d, H in cons:
        ev, V = np.linalg.eigh(H)
        L = (V * np.sqrt(np.maximum(ev, 0))).T                # LᵀL = H
        for c in range(2):
            for e in range(2):
                A[r, 2 * j + e] += L[c, e]
                A[r, 2 * i + e] -= L[c, e]
            b[r] = L[c] @ d
            r += 1
    for c in range(2):
        A[r, c] = 1e3
        b[r] = 1e3 * F[ids[0]]["T"][c, 3]
        r += 1
    x = lsqr(A.tocsr(), b, atol=1e-12, btol=1e-12, iter_lim=50000)[0].reshape(n, 2)
    T = np.array([F[k]["T"] for k in ids])
    T[:, :2, 3] = x
    dr = np.array([F[k]["T"][:2, 3] for k in ids])
    print(f"  相对 DR 位移 med {np.median(np.linalg.norm(x-dr,axis=1)):.2f} m；"
          f"路程比 {np.linalg.norm(np.diff(x,axis=0),axis=1).sum()/np.linalg.norm(np.diff(dr,axis=0),axis=1).sum():.3f}")
    np.savez(rec / a.out, ids=np.array(ids), T=T)
    return 0


if __name__ == "__main__":
    sys.exit(main())
