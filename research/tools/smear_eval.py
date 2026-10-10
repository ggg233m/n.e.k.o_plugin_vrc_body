# -*- coding: utf-8 -*-
r"""拖影快速评估：逐对墙面失配（同格法向偏移 rms），可对比多组位姿修正。

`.tmp/pc_smear_pair.py` / `.tmp/vo_direct_check.py` 的提速重写（口径一致）：
 * 分格聚合用 np.unique + bincount（原版 Python dict 逐点 append，占 83% 时间）；
 * 每帧平面拟合 / 点云 / ORB 特征只算一次并缓存到 `<rec>/.smear_cache.npz`；
 * 多场录制用进程池并行。

每对 (a,b)：a 帧最大竖直面 (n,p0) → 两帧点投到面内 0.25 m 格 → 两帧都 ≥5 点且 a 在面 ±10 cm 内的格
→ 格均值法向差的 rms。报：DR 原样 / 本对视觉 Δt 修正 / 按墙距分箱。

    .venv/Scripts/python.exe research/tools/smear_eval.py                 # 默认 4 场
    .venv/Scripts/python.exe research/tools/smear_eval.py 20261001_044153 --span 3
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "tools"))
import kf_visual_odom as V      # noqa: E402

REC_ROOT = ROOT / "navmesh_recordings"
DEFAULT_RECS = ["20261001_044153", "20261005_235232", "20261006_001019", "20261006_001523"]
CELL, MINPTS = 0.25, 5


def fit_vertical_plane(P, rng, *, thresh=0.030, iters=1200, sub=6000, zlo=0.6, zhi=5.0,
                       max_vz=0.35, min_in=1000):
    """与 .tmp/pc_smear_pair.py 同口径；返回 (n, p0, 自厚) 或 None。"""
    sel = (P[:, 0] > zlo) & (P[:, 0] < zhi)
    if int(sel.sum()) < min_in:
        return None
    Q = P[sel]
    if len(Q) > sub:
        Q = Q[rng.choice(len(Q), size=sub, replace=False)]
    tri = rng.integers(0, len(Q), size=(iters, 3))
    A, B, C = Q[tri[:, 0]], Q[tri[:, 1]], Q[tri[:, 2]]
    nn = np.cross(B - A, C - A)
    ln = np.linalg.norm(nn, axis=1)
    keep = ln > 1e-9
    nn, A = nn[keep] / ln[keep, None], A[keep]
    vert = np.abs(nn[:, 2]) < max_vz
    if not vert.any():
        return None
    nn, A = nn[vert], A[vert]
    # 打分：|Q·n − A·n| < thresh，矩阵乘一次算完
    d = np.abs(Q @ nn.T - np.einsum("ij,ij->i", A, nn)[None, :])
    bi = int(np.argmax((d < thresh).sum(0)))
    n, p0 = nn[bi].copy(), A[bi].copy()
    for _ in range(2):
        m = np.abs((P - p0) @ n) < thresh
        if int(m.sum()) < min_in:
            return None
        q = P[m]
        c0 = q.mean(0)
        _, Vv = np.linalg.eigh((q - c0).T @ (q - c0) / len(q))
        n, p0 = Vv[:, 0], c0
    m = np.abs((P - p0) @ n) < 0.10
    n = n / np.linalg.norm(n)
    if n[0] < 0:
        n = -n
    return n, p0, float(((P[m] - p0) @ n).std())


def load_cached(rec: Path):
    """每帧：pts（体素 3 cm 下采样，已足够 0.25 m 格 ≥5 点）、平面、T_dr、ORB。"""
    F = V.load(rec)
    cache = rec / ".smear_cache.npz"
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        planes = c["planes"].item()
        pts = c["pts"].item()
    else:
        rng = np.random.default_rng(0)
        planes, pts = {}, {}
        for k in F:
            P = np.load(rec / "kf" / f"{k:06d}.npz")["pts"].astype(np.float64)
            g = fit_vertical_plane(P, rng)
            if g is not None:
                planes[k] = g
            q = np.floor(P / 0.03).astype(np.int64)
            _, i = np.unique(q, axis=0, return_index=True)
            pts[k] = P[np.sort(i)].astype(np.float32)
        np.savez(cache, planes=np.array(planes, dtype=object), pts=np.array(pts, dtype=object))
    for k in F:
        F[k]["pts"] = pts[k].astype(np.float64)
        F[k]["plane"] = planes.get(k)
    return F


MEDIAN = False  # 格内用中位数代替均值（抗杂物，且不像 GATE 那样截断大偏移）
VISONLY = False  # --poses 时只评有视觉约束的对（与逐对 oracle 同一对集）
STAT = "rms"   # rms | med | tail
GATE = None    # 逐点门：两帧都只留 |到 a 面距离| < GATE 的点（None = 原口径）


def mism(PA, PB, n, p0):
    """同格法向失配 rms；向量化。"""
    e1 = np.cross(n, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    E = np.column_stack([e1, e2])

    def agg(P):
        X = P - p0
        if GATE is not None:
            X = X[np.abs(X @ n) < GATE]
        k = np.floor(X @ E / CELL).astype(np.int64)
        key = k[:, 0] * 1_000_003 + k[:, 1]
        u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        h = X @ n
        if not MEDIAN:
            return u, cnt, np.bincount(inv, weights=h) / cnt
        o = np.lexsort((h, inv))
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        hs = h[o]
        return u, cnt, 0.5 * (hs[start + (cnt - 1) // 2] + hs[start + cnt // 2])

    ua, ca, ma = agg(PA)
    ub, cb, mb = agg(PB)
    _, ia, ib = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
    ok = (ca[ia] >= MINPTS) & (cb[ib] >= MINPTS) & (np.abs(ma[ia]) < 0.1)
    if int(ok.sum()) < 3:
        return None
    dlt = mb[ib][ok] - ma[ia][ok]
    if STAT == "med":       # 整面偏移（稳健）：|中位数|
        return float(abs(np.median(dlt)))
    if STAT == "tail":      # 尾巴：|dlt|>0.3 m 的格占比
        return float(np.mean(np.abs(dlt) > 0.3))
    return float(np.sqrt(np.mean(dlt ** 2)))


def world(r, T=None):
    T = r["T"] if T is None else T
    return r["pts"] @ T[:3, :3].T + T[:3, 3]


def eval_rec(name: str, span: int = 3, gate=None, median=False, stat="rms", poses=None, maxdt=3.0, visonly=False):
    global GATE, MEDIAN, STAT, VISONLY
    GATE, MEDIAN, STAT, VISONLY = gate, median, stat, visonly
    t0 = time.perf_counter()
    rec = REC_ROOT / name
    F = load_cached(rec)
    TP = None
    if poses:
        fp = np.load(rec / poses)
        TP = {int(k): np.asarray(T, float) for k, T in zip(fp["ids"], fp["T"])}
        F = {k: v for k, v in F.items() if k in TP}
    ids = sorted(F)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(len(ids) - 1):
        a = F[ids[i]]
        if a["plane"] is None:
            continue
        nb, p0b, thick = a["plane"]
        Ta = a["T"]
        n = Ta[:3, :3] @ nb
        p0 = Ta[:3, :3] @ p0b + Ta[:3, 3]
        PA = world(a)
        cdist = float(abs((p0 - Ta[:3, 3]) @ n))
        for s in range(1, span + 1):
            if i + s >= len(ids):
                break
            b = F[ids[i + s]]
            if b["t"] - a["t"] > maxdt:
                break
            PB = world(b)
            m0 = mism(PA, PB, n, p0)
            if TP is not None and VISONLY and V.rel_translation(a, b, zmax=5, thr=0.10, min_in=10, rng=rng) is None:
                continue
            if TP is not None:       # 第二列 = 给定位姿文件（整图一致解，可部署）
                Tpa, Tpb = TP[ids[i]], TP[ids[i + s]]
                m1 = mism(world(a, Tpa), world(b, Tpb), Tpa[:3, :3] @ nb,
                          Tpa[:3, :3] @ p0b + Tpa[:3, 3])
            else:                    # 第二列 = 本对视觉 Δt 直接修（逐对 oracle，不可部署）
                r = V.rel_translation(a, b, zmax=5, thr=0.10, min_in=10, rng=rng)
                if r is None:
                    continue
                corr = np.array([*(r[0] - (b["T"][:3, 3] - Ta[:3, 3]))[:2], 0.0])
                m1 = mism(PA, PB + corr, n, p0)
            if m0 is not None and m1 is not None:
                rows.append((m0, m1, thick, cdist))
    o = np.array(rows)
    out = [f"{name}  对 {len(o)}  ({time.perf_counter()-t0:.1f} s)",
           f"  墙面失配中位: DR {np.median(o[:,0])*100:.1f} → {'位姿' if TP is not None else '视觉Δt'} {np.median(o[:,1])*100:.1f} cm；"
           f"自厚 {np.median(o[:,2])*100:.1f}；改善占 {100*np.mean(o[:,1]<o[:,0]):.0f}%"]
    for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 6)):
        k = (o[:, 3] >= lo) & (o[:, 3] < hi)
        if k.sum() >= 5:
            out.append(f"  墙距 {lo}–{hi} m (n={k.sum():3d}): DR {np.median(o[k,0])*100:5.1f} → "
                       f"修后 {np.median(o[k,1])*100:5.1f} cm；自厚 {np.median(o[k,2])*100:.1f}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="*", default=DEFAULT_RECS)
    ap.add_argument("--span", type=int, default=3)
    ap.add_argument("--gate", type=float, default=None,
                    help="逐点门（m）：两帧只留到 a 墙面 ±gate 内的点，排除格内杂物")
    ap.add_argument("--stat", choices=["rms", "med", "tail"], default="rms",
                    help="每对汇总：rms（原口径）/ med（整面偏移 |中位|）/ tail（|Δ|>0.3 m 格占比）")
    ap.add_argument("--poses", default=None, help="<rec>/ 下的位姿 npz（ids,T），对比 DR")
    ap.add_argument("--vis-only", action="store_true", help="--poses 时只评能做视觉匹配的对")
    ap.add_argument("--maxdt", type=float, default=3.0, help="配对的最大时间差 s")
    ap.add_argument("--median", action="store_true", help="格内取中位数（抗杂物、不截断）")
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=min(4, len(a.recs))) as ex:
        for s in ex.map(eval_rec, a.recs, [a.span] * len(a.recs), [a.gate] * len(a.recs), [a.median] * len(a.recs), [a.stat] * len(a.recs),
                          [a.poses] * len(a.recs), [a.maxdt] * len(a.recs), [a.vis_only] * len(a.recs)):
            print(s, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
