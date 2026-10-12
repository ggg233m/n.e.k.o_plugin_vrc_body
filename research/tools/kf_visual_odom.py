# -*- coding: utf-8 -*-
r"""关键帧间视觉平移约束（原型）：用相邻关键帧的 ORB 3D-3D 匹配修航位推算的帧间平移。

背景：`Docs/点云拖影成因定案-跨帧位姿不一致（2026-10-08）.md` §七 —— 拖影跨场稳定存在，
根因是建图按 DR 位姿叠关键帧、帧间没有几何对齐。

思路：
 * 朝向来自 HMD、不漂 ⇒ 旋转已知，只估平移（3 DOF）。
 * 用 `kf/<id>.npz` 里已有的 `xyz`/`des3d`（双目三角化的 ORB 点，头部 base 系）做 **3D-3D 描述子匹配**。
   与稠密 ICP 不同：描述子匹配给的是**点对点**对应，沿墙滑动会让对应点错开 ⇒ 不存在"长直墙退化"
   （那正是 `Docs/archive/稠密配准约束源-证伪（2026-10-06）.md` 的失败模式）。
 * 每个匹配给出一个 Δt 估计：t_b − t_a = R_a·x_a − R_b·x_b；RANSAC（单点假设）+ 按深度方差加权。
 * 平移位姿图：DR 里程（相邻帧，弱权）+ 视觉约束（k→k+1..k+S），线性最小二乘，锚定首帧。

产出 `<rec>/vo_poses.npz`（ids, T），旋转原样取 T_dr。

    .venv/Scripts/python.exe research/tools/kf_visual_odom.py 20261001_044153
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

ROOT = Path(__file__).resolve().parents[2]
REC_ROOT = ROOT / "navmesh_recordings"

FB = 202.5 * 0.126          # f·b（像素·追踪米）
DD = 0.3                    # 视差噪声 px（FACTS §三：0.2–0.3）


def load(rec: Path):
    ev = [json.loads(x) for x in (rec / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    kt = {int(e["k"]): float(e["t"]) for e in ev if e.get("kind") == "kf"}
    F = {}
    for p in sorted((rec / "kf").glob("*.npz"), key=lambda q: int(q.stem)):
        k = int(p.stem)
        if k not in kt:
            continue
        d = np.load(p)
        F[k] = {"T": np.asarray(d["T_dr"], float), "xyz": d["xyz"].astype(np.float64),
                "des": d["des3d"], "t": kt[k]}
    return F


def match(a, b, ratio=0.8):
    if len(a["des"]) < 2 or len(b["des"]) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    f = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(a["des"], b["des"], k=2)
         if len(m) == 2 and m[0].distance < ratio * m[1].distance}
    r = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(b["des"], a["des"], k=2)
         if len(m) == 2 and m[0].distance < ratio * m[1].distance}
    return np.array([(i, j) for i, j in f.items() if r.get(j) == i], int).reshape(-1, 2)


def match_guided(a, b, *, radius=0.8, max_ham=64):
    """DR 预测引导匹配：b 的点按 DR 位姿变到世界系，只在 a 的世界点 radius 内找描述子最近者（互为最近）。
    DR 帧间误差 ~0.2–0.5 m ⇒ radius 0.8 m 足够；比全局比值测试多保留重复纹理处的匹配。"""
    if len(a["des"]) == 0 or len(b["des"]) == 0:
        return np.zeros((0, 2), int)
    Wa = a["xyz"] @ a["T"][:3, :3].T + a["T"][:3, 3]
    Wb = b["xyz"] @ b["T"][:3, :3].T + b["T"][:3, 3]
    D = np.linalg.norm(Wa[:, None, :] - Wb[None, :, :], axis=2)
    ham = np.unpackbits(a["des"][:, None, :] ^ b["des"][None, :, :], axis=2).sum(2).astype(float)
    ham[D > radius] = 1e9
    ia = ham.argmin(1)
    ib = ham.argmin(0)
    out = [(i, j) for i, j in enumerate(ia) if ib[j] == i and ham[i, j] <= max_ham]
    return np.array(out, int).reshape(-1, 2)


def rel_translation(a, b, *, zmax, thr, min_in, rng, guided=False):
    """返回 (Δt=t_b−t_a 世界系, 信息矩阵 3×3, 内点数) 或 None。"""
    m = match_guided(a, b) if guided else match(a, b)
    if len(m) < min_in:
        return None
    xa, xb = a["xyz"][m[:, 0]], b["xyz"][m[:, 1]]
    za, zb = xa[:, 0], xb[:, 0]                      # base 系 x = 前向深度
    ok = (za > 0.3) & (zb > 0.3) & (za < zmax) & (zb < zmax)
    if int(ok.sum()) < min_in:
        return None
    xa, xb, za, zb = xa[ok], xb[ok], za[ok], zb[ok]
    Ra, Rb = a["T"][:3, :3], b["T"][:3, :3]
    est = xa @ Ra.T - xb @ Rb.T                      # 每个匹配的 Δt 估计
    # 深度噪声沿视线：σ_z = z²/(f b)·δd；横向远小，取 σ_z/5
    sa, sb = za ** 2 / FB * DD, zb ** 2 / FB * DD
    # RANSAC：单点假设
    best = None
    cand = rng.choice(len(est), size=min(len(est), 80), replace=False)
    for i in cand:
        inl = np.linalg.norm(est - est[i], axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if int(best.sum()) < min_in:
        return None
    # 加权：每点协方差 = 两帧沿视线方向的各向异性噪声之和
    info = np.zeros((3, 3))
    rhs = np.zeros(3)
    for k in np.flatnonzero(best):
        C = np.zeros((3, 3))
        for R, x, s in ((Ra, xa[k], sa[k]), (Rb, xb[k], sb[k])):
            u = R @ (x / max(np.linalg.norm(x), 1e-9))
            C += s ** 2 * np.outer(u, u) + (s / 5) ** 2 * (np.eye(3) - np.outer(u, u))
        W = np.linalg.inv(C + 1e-6 * np.eye(3))
        info += W
        rhs += W @ est[k]
    d = np.linalg.solve(info, rhs)
    # 内点自洽二次过滤
    res = np.linalg.norm(est[best] - d, axis=1)
    if np.median(res) > thr:
        return None
    return d, info, int(best.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--span", type=int, default=3, help="每帧向后连几帧")
    ap.add_argument("--zmax", type=float, default=4.0)
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--min-in", type=int, default=12)
    ap.add_argument("--dr-sigma", type=float, default=0.15, help="DR 帧间平移噪声 σ（追踪米/步）")
    ap.add_argument("--max-corr", type=float, default=0.8, help="视觉与 DR 差超过此值的约束丢弃")
    ap.add_argument("--guided", action="store_true", help="DR 预测引导匹配")
    ap.add_argument("--k-osc", type=float, default=1.0, help="DR 帧间平移先乘此系数（同 osc_speed_scale）")
    ap.add_argument("--out", default="vo_poses.npz")
    a = ap.parse_args()
    rec = REC_ROOT / a.rec
    F = load(rec)
    ids = sorted(F)
    if a.k_osc != 1.0:             # 按系数重放 DR 轨迹（旋转不变，只缩帧间平移）
        p0, acc = F[ids[0]]["T"][:3, 3].copy(), F[ids[0]]["T"][:3, 3].copy()
        for k in ids:
            raw = F[k]["T"][:3, 3].copy()
            acc = acc + (raw - p0) * a.k_osc
            p0 = raw
            F[k]["T"] = F[k]["T"].copy()
            F[k]["T"][:3, 3] = acc
    idx = {k: i for i, k in enumerate(ids)}
    rng = np.random.default_rng(0)

    cons = []                      # (i, j, d, info)
    diffs, n_try = [], 0
    for i, k in enumerate(ids):
        for s in range(1, a.span + 1):
            if i + s >= len(ids):
                break
            kb = ids[i + s]
            if F[kb]["t"] - F[k]["t"] > 3.0:
                break
            n_try += 1
            r = rel_translation(F[k], F[kb], zmax=a.zmax, thr=a.thr, min_in=a.min_in, rng=rng, guided=a.guided)
            if r is None:
                continue
            d, info, nin = r
            d_dr = F[kb]["T"][:3, 3] - F[k]["T"][:3, 3]
            gap = float(np.linalg.norm((d - d_dr)[:2]))
            if gap > a.max_corr:
                continue
            d[2] = d_dr[2]                     # 地图 z 恒 0（DR 只有平面），视觉 z 不用
            cons.append((i, i + s, d, info[:2, :2]))
            diffs.append(gap)
    print(f"[{a.rec}] 关键帧 {len(ids)}；尝试 {n_try} 对，接受 {len(cons)}（{100*len(cons)/max(n_try,1):.0f}%）")
    if diffs:
        dd = np.array(diffs)
        print(f"  |视觉 − DR| 帧间平移差：med {np.median(dd)*100:.1f} cm  p90 {np.percentile(dd,90)*100:.1f} cm")

    # 平移位姿图（只解 xy）：DR 相邻步 + 视觉约束
    n = len(ids)
    rows = (n - 1) * 2 + len(cons) * 2 + 2
    A = lil_matrix((rows, 2 * n))
    b = np.zeros(rows)
    r = 0
    w_dr = 1.0 / a.dr_sigma
    for i in range(n - 1):
        d_dr = F[ids[i + 1]]["T"][:2, 3] - F[ids[i]]["T"][:2, 3]
        for c in range(2):
            A[r, 2 * (i + 1) + c] = w_dr
            A[r, 2 * i + c] = -w_dr
            b[r] = w_dr * d_dr[c]
            r += 1
    for i, j, d, I2 in cons:
        L = np.linalg.cholesky(I2 + 1e-9 * np.eye(2)).T      # LᵀL = I
        for c in range(2):
            for e in range(2):
                A[r, 2 * j + e] += L[c, e]
                A[r, 2 * i + e] -= L[c, e]
            b[r] = L[c] @ d[:2]
            r += 1
    for c in range(2):                     # 锚定首帧
        A[r, c] = 1e3
        b[r] = 1e3 * F[ids[0]]["T"][c, 3]
        r += 1
    x = lsqr(A.tocsr(), b, atol=1e-10, btol=1e-10, iter_lim=20000)[0].reshape(n, 2)

    T_out = []
    for i, k in enumerate(ids):
        T = F[k]["T"].copy()
        T[:2, 3] = x[i]
        T_out.append(T)
    T_out = np.array(T_out)
    shift = np.linalg.norm(T_out[:, :2, 3] - np.array([F[k]["T"][:2, 3] for k in ids]), axis=1)
    step_dr = np.linalg.norm(np.diff(np.array([F[k]["T"][:2, 3] for k in ids]), axis=0), axis=1)
    step_vo = np.linalg.norm(np.diff(T_out[:, :2, 3], axis=0), axis=1)
    print(f"  相对 DR 的位移：med {np.median(shift):.2f} m  max {shift.max():.2f} m")
    print(f"  路程：DR {step_dr.sum():.1f} m  修正后 {step_vo.sum():.1f} m"
          f"（比 {step_vo.sum()/max(step_dr.sum(),1e-9):.3f}）")
    np.savez(rec / a.out, ids=np.array(ids), T=T_out)
    print(f"  写出 {rec / a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
