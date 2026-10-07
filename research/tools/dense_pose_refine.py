# -*- coding: utf-8 -*-
r"""把**稠密点云配准**当新约束源，喂进平移位姿图，看能不能压过现在的回环位姿。

背景（2026-10-06，`Docs/稠密配准约束源-证伪（2026-10-06）.md`）
----------------------------------------------------------------
* 回环是唯一有效修正源，但供给被 `min_inliers=50` 卡死（每帧稀疏特征仅 188–729 个）。
* 每关键帧另有 ~4.4 万**稠密双目点**（基线 0.126、fx 已知 ⇒ 绝对尺度），位姿优化从未用过。
* yaw 来自 HMD、不漂 ⇒ **旋转是已知的**，所以这里做**纯平移**配准（3-DOF），不是 6-DOF ICP。
  这比全 6-DOF 更受约束：没有旋转可滑，唯一自由度是平移。

验收（三件套，见同日「分段形变证伪」的教训：**拟合残差一律不可用**）
------------------------------------------------------------------
1. **几何**：重访处稠密云 NN 距离 —— 且必须**留出该对**再拟合（否则自己验自己）；
2. **长度**：优化后轨迹总长 / DR 总长（≈OSC 口径），大幅膨胀 = 撕裂；
3. **连续**：相邻帧步长相对 DR 的变化，出现跳变 = 块界/段界断裂。

🔴 **2026-10-06 结案：本路线已证伪出局**（权威=`Docs/稠密配准约束源-证伪（2026-10-06）.md`）。
配准残差能贴到密度地板（0.138 m）、内点 44–79%，**但残差好 ≠ 平移对**：在 DR 可当真值的
短跨度（相隔 5 帧）上它错 **1.88 m** 中位、且系统性缩水（|d_ICP|/|d_DR| = 0.53）。
实现本身没问题（见 `dense_icp_selftest.py`，合成平移误差 0.06 m）。
本文件保留为**实验台**：默认闸门（滑动检测）会把这类约束全部拒掉，行为正确；
再试稠密配准前请先看文档 §七（要先解决低重叠）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[2]   # research/tools/ → 仓库根
WORLD = "wrld_home-7cf435ea"


# ────────────────────────────── 数据 ──────────────────────────────
def pick_rec(sid: str) -> Path:
    """会话 id 与录制目录名不同名（如 235237 ↔ 235232）：按时间戳就近匹配。"""
    cands = [p for p in (ROOT / "navmesh_recordings").iterdir()
             if p.is_dir() and (p / "kf").is_dir() and p.name[:8] == sid[:8]]
    if not cands:
        raise SystemExit(f"找不到 {sid} 对应的录制目录")
    return min(cands, key=lambda p: abs(int(p.name[9:]) - int(sid[9:])))


class Session:
    def __init__(self, sid: str) -> None:
        self.sid = sid
        self.rec = pick_rec(sid)
        z = np.load(ROOT / "navmesh_memory" / WORLD / "sessions" / sid / "poses.npz")
        d = np.asarray(z["dist_m"], np.float64)
        o = np.argsort(d)
        self.ids = np.asarray(z["ids"])[o]
        self.dist = d[o]
        self.T_map = np.asarray(z["T_map"], np.float64)[o]
        self.T_dr = np.asarray(z["T_dr"], np.float64)[o]
        self.n = len(self.ids)

    @property
    def R(self) -> np.ndarray:
        return self.T_map[:, :3, :3]          # 旋转来自 HMD，不随路程漂移

    @property
    def p_map(self) -> np.ndarray:
        return self.T_map[:, :3, 3].copy()

    @property
    def p_dr(self) -> np.ndarray:
        return self.T_dr[:, :3, 3].copy()

    def thumbs(self) -> np.ndarray:
        """缩略图外观描述子 (n, D)：**与任何位姿源都无关**的候选选择器。

        用 DR 位置挑"重访对"会把裁判偏向 DR（DR 说近就选进来 ⇒ DR 天然占优）。
        外观相似度只看画面，配合 yaw（来自 HMD，两种位姿源共用、中立）⇒ 无偏。
        """
        from PIL import Image
        vecs = np.zeros((self.n, 40 * 24), np.float32)
        have = np.zeros(self.n, bool)
        for i, k in enumerate(self.ids):
            f = ROOT / "navmesh_memory" / WORLD / "sessions" / self.sid / "thumb" / f"{k + 1:06d}.jpg"
            if not f.exists():
                continue
            g = np.asarray(Image.open(f).convert("L").resize((40, 24)), np.float32)
            g = (g - g.mean()) / (g.std() + 1e-6)
            vecs[i] = g.ravel()
            have[i] = True
        return vecs, have

    def cloud(self, i: int, nper: int, rng: np.random.Generator, vox: float = 0.05
              ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (相机系点, 旋转到世界朝向的点 q)。世界点 = q + t_i。"""
        with np.load(self.rec / "kf" / f"{self.ids[i]:06d}.npz") as kf:
            p = np.asarray(kf["pts"], np.float64)
        p = p[np.isfinite(p).all(1)]
        p = p[(np.abs(p[:, 0]) < 12) & (np.abs(p[:, 1]) < 12) & (np.abs(p[:, 2]) < 20)]
        if len(p) > nper:
            p = p[rng.choice(len(p), nper, replace=False)]
        if vox > 0:
            _, uniq = np.unique(np.floor(p / vox).astype(np.int32), axis=0, return_index=True)
            p = p[uniq]
        return p, p @ self.R[i].T


# ────────────────────────── 纯平移稠密配准 ──────────────────────────
def voxelize(p: np.ndarray, v: float) -> np.ndarray:
    if v <= 0:
        return p
    _, u = np.unique(np.floor(p / v).astype(np.int32), axis=0, return_index=True)
    return p[u]


def _icp_one(A: np.ndarray, B: np.ndarray, d0: np.ndarray, *, thr0: float,
             iters: int, robust_c: float) -> tuple[np.ndarray, float, float]:
    tree = cKDTree(B)
    d = np.asarray(d0, np.float64).copy()
    rmse, thr, keep_frac = np.inf, float(thr0), 0.0
    for _ in range(iters):
        P = A + d
        dist, idx = tree.query(P, workers=-1)
        if np.isfinite(rmse):
            thr = float(np.clip(3.0 * rmse, 0.10, thr0))
        keep = dist < thr
        keep_frac = float(keep.mean())
        if int(keep.sum()) < 50:
            break
        r = P[keep] - B[idx[keep]]
        rr = np.hypot(r[:, 0], r[:, 1])            # 水平残差（竖直方向由重力锚定，不稳）
        w = 1.0 / (1.0 + (rr / robust_c) ** 2)
        delta = -np.average(r, axis=0, weights=w)
        d = d + delta
        rmse = float(np.sqrt(np.average((r ** 2).sum(1), weights=w)))
        if float(np.linalg.norm(delta)) < 1e-4:
            break
    return d, rmse, keep_frac


def icp_translation(A: np.ndarray, B: np.ndarray, d0: np.ndarray, *,
                    iters: int = 40, robust_c: float = 0.30,
                    pyramid: tuple[float, ...] = (0.40, 0.15, 0.0)) -> dict:
    """对齐两片**已旋转到世界朝向**的云：求 d 使 A + d ≈ B。

    * ``A``/``B``：世界朝向的局部云（q_i = R_i @ p_i），世界点 = q_i + t_i。**调用方须传
      A = j 的云、B = i 的云** ⇒ 解出的 ``d = t_j − t_i``（与位姿图的边 z 同向）。
      🔴 符号坑（踩过）：若传 A=i、B=j，解出的是 ``t_i − t_j``（反号），直接拿去跟
      DR/T_map 比或当图边，会把"两倍相对平移"当成"分歧"，整条链全错。
    * ``d0``：初值（DR 给的 ``t_j − t_i``）；
    * ``pyramid``：**粗到细**。DR 初值偏差可能 1 m+，直接用细云配不上（对应全被拒、
      落不进收敛盆地）⇒ 先在 0.4 m 体素上粗配拿到盆地，再逐级细化。
    * 返回 d（= t_b − t_a）、残差、内点比例。
    """
    d = np.asarray(d0, np.float64).copy()
    rmse, keep = np.inf, 0.0
    for v in pyramid:
        thr0 = max(3.0 * v, 0.60) if v > 0 else 0.60
        d, rmse, keep = _icp_one(voxelize(A, v), voxelize(B, v), d,
                                 thr0=thr0, iters=iters, robust_c=robust_c)
    return {"d": d, "rmse": rmse, "keep": keep}


def constraint_gain(A: np.ndarray, B: np.ndarray, d: np.ndarray, d0: np.ndarray, *,
                    step: float = 0.5, rng: np.random.Generator | None = None
                    ) -> tuple[float, float]:
    """滑动检测：在解处**冻结对应**，挪 step 米，残差涨几倍？

    * 沿**修正方向**（解相对 DR 初值偏离的方向）挪：涨得少 ⇒ 平行墙走廊类弱约束 ⇒ 不可信；
    * 同时测 3 个**随机方向**作基线：修正方向若比随机方向还平 ⇒ 它正是最弱的方向 ⇒ 滑动。

    返回 (修正方向涨幅, 随机方向涨幅中位)。
    """
    rng = rng or np.random.default_rng(0)

    def resid(shift: np.ndarray) -> float:
        """**trimmed-ICP 代价**：最近距离里取最优 40% 求 RMS。

        🔴 踩过的坑（勿复用）：用「cap 1.0 的截断均值」或「先滤 dist<1.0 再取中位」都会
        被**没有对应的那一半点**稀释 ⇒ 残差剖面看起来是平的 ⇒ 误判成"弱约束/滑动"。
        实测对照（同批对）：截断均值下修正方向 ×0.85–0.96（假平），trimmed RMS 下
        视线轴剖面 1.53→0.64→**0.10**→0.62→1.50（尖谷，谷宽 0.00 m）。
        """
        P = A + d + shift
        dist, _ = cKDTree(B).query(P, workers=-1)
        dist = np.sort(dist)
        k = max(50, int(0.40 * len(dist)))
        return float(np.sqrt((dist[:k] ** 2).mean()))

    r0 = resid(np.zeros(3))
    vec = np.asarray(d, np.float64) - np.asarray(d0, np.float64)
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-6 or not np.isfinite(r0) or r0 < 1e-6:
        return 0.0, 0.0
    g_dev = resid(vec / nrm * step) / r0
    rs = []
    for _ in range(3):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        rs.append(resid(u * step) / r0)
    return float(g_dev), float(np.median(rs))


# ──────────────────────────── 位姿图求解 ────────────────────────────
def solve_graph(n: int, oi: np.ndarray, oj: np.ndarray, oz: np.ndarray, ow: np.ndarray,
                li: np.ndarray, lj: np.ndarray, lz: np.ndarray, lw: np.ndarray,
                x0: np.ndarray, iters: int = 6, robust_k: float = 2.0,
                lsig: np.ndarray | None = None) -> np.ndarray:
    """min Σ w‖x_j − x_i − z‖²（3D，首节点固定），回环边带 Cauchy IRLS。

    里程边 + 回环边一次性建图拉普拉斯；回环权重按残差重估（与 `nav_loop.optimize` 同构）。
    """
    lsig = np.ones(len(li)) if lsig is None else np.asarray(lsig, np.float64)

    def build(w_loop: np.ndarray) -> np.ndarray:
        I = np.concatenate([oi, li])
        J = np.concatenate([oj, lj])
        Z = np.concatenate([oz, lz])                      # (m, 3)
        W = np.concatenate([ow, np.maximum(w_loop, 1e-6)])
        g = np.zeros((n, 3))
        for k in range(3):
            wz = W * Z[:, k]
            g[:, k] = np.bincount(J, wz, minlength=n) - np.bincount(I, wz, minlength=n)
        rows = np.concatenate([I, J, I, J])
        cols = np.concatenate([I, J, J, I])
        vals = np.concatenate([W, W, -W, -W])
        keep = rows != 0
        rows = np.append(rows[keep], 0)
        cols = np.append(cols[keep], 0)
        vals = np.append(vals[keep], 1.0)
        g[0] = x0
        H = csc_matrix((vals, (rows, cols)), shape=(n, n))
        return splu(H).solve(g)

    wl = np.ones(len(li))
    x = build(wl)
    for _ in range(iters):
        if not len(li):
            break
        r = np.linalg.norm(x[lj] - x[li] - lz, axis=1)
        wn = 1.0 / (1.0 + (r / (robust_k * lsig)) ** 2)
        if np.all(np.abs(wn - wl) < 1e-3):
            break
        wl = wn
        x = build(wl)
    return x


# ────────────────────────────── 主流程 ──────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sid", nargs="?", default="20261005_235237")
    ap.add_argument("--np", type=int, default=2500, help="每片云抽样点数")
    ap.add_argument("--max-pairs", type=int, default=40)
    ap.add_argument("--pos-m", type=float, default=1.5, help="DR 位置差上限（候选）")
    ap.add_argument("--yaw-deg", type=float, default=15.0, help="朝向差上限")
    ap.add_argument("--min-path", type=float, default=25.0, help="里程差下限")
    ap.add_argument("--select", choices=("dr", "appearance", "both"), default="appearance",
                    help="候选口径：dr 偏向 DR（对照）；appearance 与位姿无关（无偏）")
    ap.add_argument("--gain", type=float, default=1.5, help="滑动检测：残差涨幅下限")
    ap.add_argument("--odo-sigma0", type=float, default=0.15)
    ap.add_argument("--odo-frac", type=float, default=0.02)
    ap.add_argument("--loop-sigma", type=float, default=0.10)
    ap.add_argument("--verbose", action="store_true", help="逐对打印配准明细")
    ap.add_argument("--save", action="store_true", help="导出优化后位姿 npz")
    a = ap.parse_args()

    S = Session(a.sid)
    rng = np.random.default_rng(7)
    print(f"会话 {a.sid} · 关键帧 {S.n} · 录制 {S.rec.name} · OSC 里程 {S.dist[-1]:.0f} m")

    dR = np.array([np.abs(np.degrees(np.arctan2(S.R[i + 1][1, 0], S.R[i + 1][0, 0])
                                     - np.arctan2(S.R[i][1, 0], S.R[i][0, 0])))
                   for i in range(S.n - 1)])
    yaw = np.concatenate([[0.0], np.cumsum(np.where(dR > 180, 360 - dR, dR))])
    pdr, pmp = S.p_dr, S.p_map

    # ① 候选：里程远 + 朝向近（yaw 来自 HMD，两种位姿源共用 ⇒ 中立）+ 第三种口径
    #    · dr         ：DR 位置近 —— **偏向 DR**，只是对照；
    #    · appearance ：缩略图外观相似 —— 与位姿无关，**无偏**；
    #    · both       ：两者都要（交集，最严）。
    yaw_ok = (np.abs(yaw[:, None] - yaw[None, :]) % 360)
    yaw_ok = np.minimum(yaw_ok, 360 - yaw_ok) < a.yaw_deg
    dpath_all = np.abs(S.dist[:, None] - S.dist[None, :]) > a.min_path
    gate = yaw_ok & dpath_all & ~np.eye(S.n, dtype=bool)

    pr = np.asarray(np.argwhere(np.triu(gate)))
    if a.select in ("dr", "both"):
        pxy = pdr[:, :2]
        close = np.linalg.norm(pxy[pr[:, 0]] - pxy[pr[:, 1]], axis=1) < a.pos_m
        if a.select == "dr":
            pr = pr[close]
        else:
            pr = pr[close]
    if a.select in ("appearance", "both"):
        vecs, have = S.thumbs()
        v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        sim = (v[pr[:, 0]] * v[pr[:, 1]]).sum(1)
        ok_have = have[pr[:, 0]] & have[pr[:, 1]]
        pr = pr[ok_have]
        sim = sim[ok_have]
        order = np.argsort(sim)[::-1]
        pr = pr[order][:max(a.max_pairs * 4, 40)]
        if a.select == "both":
            close = np.linalg.norm(pdr[pr[:, 0], :2] - pdr[pr[:, 1], :2], axis=1) < a.pos_m
            pr = pr[close]
    if len(pr) > a.max_pairs:
        pr = pr[:a.max_pairs]
    print(f"候选重访对 {len(pr)} 组（口径={a.select}、朝向差<{a.yaw_deg}°、里程差>{a.min_path} m）")

    # ② 逐对纯平移配准
    cons: list[dict] = []
    for i, j in pr:
        _, Bj = S.cloud(int(i), a.np, rng)      # B = i 的云（被配准的目标）
        _, A = S.cloud(int(j), a.np, rng)       # A = j 的云（要挪的那片）
        B = Bj
        d0 = pdr[int(j)] - pdr[int(i)]          # = t_j − t_i
        r = icp_translation(A, B, d0)
        g, g_rand = constraint_gain(A, B, r["d"], d0, rng=rng)
        cons.append({"i": int(i), "j": int(j), **r, "gain": g, "gain_rand": g_rand,
                     "corr": float(np.linalg.norm(r["d"] - d0)),
                     # ICP 当裁判的正面对决：两种位姿源与该 ICP 测量差多少（同一对、同一测量）
                     "corr_map": float(np.linalg.norm(r["d"] - (pmp[int(j)] - pmp[int(i)])))})
    if a.gain > 0:
        ok = [c for c in cons if c["rmse"] < 0.40 and c["keep"] > 0.15
              and c["gain"] >= a.gain and c["gain"] >= 0.9 * c["gain_rand"]]
        why = f"（残差<0.40、内点>15%、滑动涨幅≥{a.gain}× 且不弱于随机方向）"
    else:
        ok = [c for c in cons if c["rmse"] < 0.40 and c["keep"] > 0.15]
        why = "（--gain 0：只卡残差<0.40、内点>15%，**滑动检测关闭**）"
    print(f"配准收敛 {len(cons)} 组 → 过闸 {len(ok)} 组{why}")
    if a.verbose:
        print(f"  {'i':>5}{'j':>6}{'残差':>7}{'内点':>6}{'修正m':>7}{'涨幅':>7}{'随机':>7}"
              f"{'|d−d_DR|':>9}{'|d−d_Tmap|':>10}")
        for c in cons:
            print(f"  {c['i']:>5}{c['j']:>6}{c['rmse']:>7.3f}{c['keep']:>6.0%}"
                  f"{c['corr']:>7.2f}{c['gain']:>7.2f}{c['gain_rand']:>7.2f}"
                  f"{c['corr_map']:>9.2f}")
    if cons:
        print(f"  残差中位 {np.median([c['rmse'] for c in cons]):.3f} m · "
              f"滑动涨幅中位 {np.median([c['gain'] for c in cons]):.2f}× "
              f"(随机方向 {np.median([c['gain_rand'] for c in cons]):.2f}×)")
        print(f"  ICP 当裁判（同一对、同一测量）：|d_ICP − d_DR| 中位 "
              f"{np.median([c['corr'] for c in cons]):.2f} m · "
              f"|d_ICP − d_Tmap| 中位 {np.median([c['corr_map'] for c in cons]):.2f} m")
    if not ok:
        print("无可用约束，退出")
        return 1

    # ③ 位姿图：里程边（DR）+ 回环边（ICP）
    idx = np.arange(S.n - 1)
    oi, oj = idx, idx + 1
    oz = pdr[1:] - pdr[:-1]
    ostep = np.linalg.norm(oz, axis=1)
    osig = a.odo_sigma0 + a.odo_frac * ostep
    ow = 1.0 / (osig ** 2)
    li = np.array([c["i"] for c in ok], int)
    lj = np.array([c["j"] for c in ok], int)
    lz = np.array([c["d"] for c in ok], float)
    lsig = np.maximum(np.array([c["rmse"] for c in ok], float), a.loop_sigma)

    def fit(exclude: int = -1) -> np.ndarray:
        if exclude >= 0:
            m = np.arange(len(li)) != exclude
            return solve_graph(S.n, oi, oj, oz, ow, li[m], lj[m], lz[m], lsig[m], pdr[0])
        return solve_graph(S.n, oi, oj, oz, ow, li, lj, lz, lsig, pdr[0])

    x_full = fit()

    # ④′ 互洽性检验：**ICP 约束之间**是否互相自洽（不需要真值）。
    #   若它们互相矛盾 ⇒ 各自都是"碰巧配上的假对齐"；若互相自洽 ⇒ 是真实几何测量。
    res = np.linalg.norm(x_full[lj] - x_full[li] - lz, axis=1)
    print(f"\n验收 0 · ICP 约束互洽性（拟合后残差）：中位 {np.median(res):.3f} m · "
          f"p90 {np.percentile(res, 90):.3f} m · 最大 {res.max():.3f} m · "
          f">0.5 m 占 {np.mean(res > 0.5):.0%}")
    off = np.linalg.norm(x_full - pdr, axis=1)
    print(f"        相对 DR 的位移：中位 {np.median(off):.2f} m · 最大 {off.max():.2f} m"
          f" · 相对 T_map：中位 {np.median(np.linalg.norm(x_full - pmp, axis=1)):.2f} m")

    # ④ 验收 1：重访处稠密云 NN 距离（**留出该对**再拟合 —— 否则自己验自己）
    rows = []
    for k, c in enumerate(ok):
        _, A = S.cloud(c["i"], 3000, rng)
        _, B = S.cloud(c["j"], 3000, rng)
        x_loo = fit(k)

        def judge(t) -> tuple[float, float]:
            dd, _ = cKDTree(B + t[c["j"]]).query(A + t[c["i"]], workers=-1)
            dd = np.sort(dd)
            kk = max(50, int(0.40 * len(dd)))
            return float(np.sqrt((dd[:kk] ** 2).mean())), float(np.median(dd))

        v_ref, r_ref = judge(x_loo)
        v_map, _ = judge(pmp)
        v_dr, _ = judge(pdr)
        rows.append((c["i"], c["j"], v_dr, v_map, v_ref, r_ref))

    print("\n验收 1 · 重访处稠密云 trimmed RMS（最优40%，m；**越低越几何自洽**；留出该对拟合）")
    print(f"{'i':>5}{'j':>6}{'DR':>9}{'T_map':>9}{'ICP优化':>10}{'(中位)':>9}")
    for r in rows:
        print(f"{r[0]:>5}{r[1]:>6}{r[2]:>9.3f}{r[3]:>9.3f}{r[4]:>10.3f}{r[5]:>9.3f}")
    med = np.median(np.array(rows)[:, 2:5], axis=0)
    print(f"{'中位':>11}{med[0]:>9.3f}{med[1]:>9.3f}{med[2]:>10.3f}")

    # ⑤ 验收 2/3：长度保持率 + 连续性
    step_ref = np.linalg.norm(x_full[1:] - x_full[:-1], axis=1)
    step_map = np.linalg.norm(pmp[1:] - pmp[:-1], axis=1)
    osc_total = float(S.dist[-1])
    print(f"\n验收 2 · 轨迹总长：OSC {osc_total:.0f} m · DR {ostep.sum():.0f} m · "
          f"T_map {step_map.sum():.0f} m · ICP优化 {step_ref.sum():.0f} m"
          f"  ⇒ 保持率 ICP {step_ref.sum() / max(ostep.sum(), 1e-9):.2f}× / "
          f"T_map {step_map.sum() / max(ostep.sum(), 1e-9):.2f}×")
    jump = np.abs(step_ref - ostep)
    print(f"验收 3 · 相邻步长相对 DR 的变化：中位 {np.median(jump):.3f} m · "
          f"p95 {np.percentile(jump, 95):.3f} m · 最大 {jump.max():.3f} m"
          f"（T_map 对照：中位 {np.median(np.abs(step_map - ostep)):.3f} m）")

    if a.save:
        out = ROOT / ".tmp" / f"refined_{a.sid}.npz"
        out.parent.mkdir(exist_ok=True)
        T = S.T_map.copy()
        T[:, :3, 3] = x_full
        np.savez(out, ids=S.ids, T_refined=T, T_map=S.T_map, T_dr=S.T_dr, dist_m=S.dist)
        print(f"\n已导出 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
