# -*- coding: utf-8 -*-
r"""稠密跨经过对齐 —— 可行性探针（只回答"能不能匹配上、能不能给出可信平移"）。

背景：`Docs/点云拖影修复原型与栅格影响（2026-10-09）.md` §九 —— 稀疏 ORB 的跨经过
成功率只有 1–6%，撑不起约束。本探针换**稠密纹理匹配**回答同一问题。

为什么这次可能行（与已证伪的稠密 ICP 的区别）：
  `Docs/archive/稠密配准约束源-证伪（2026-10-06）.md` 的失败模式是"长直墙 ⇒ 沿墙平移
  不改残差 ⇒ 残差漂亮但平移错"。本探针**不用残差最小化**，用的是**点对点对应**：
  纹理匹配给出"图里这个点 ↔ 图里那个点"，沿墙滑动会让对应点错开 ⇒ 不存在那个退化。

⚠️ 两个已核实的实现要点（踩过，别再犯）：
  1. **数据约定**：`pts` 与 `stereo_points` 输出**逐位一致**，是左目光学系 `[z,-x,-y]`，
     故 `u = cx - pts[:,1]*fx/z`、`v = cy - pts[:,2]*fy/z`
     （已用两种独立判据验证：视差有效性 100%、隐式视差与 SGBM 逐点一致 0.000 px）。
  2. **不能只在纹理角点上查深度**：`pts` 是 `step=2` 的网格采样且**有空洞**
     （10 px 网格覆盖只有 49–83%），在 Shi-Tomasi 角点处查 KD-tree 会大量落空。

🔴 **本次实测结论（2026-10-10）：主项是深度精度，次项是误匹配。**
  * **有效的判据是"同深度档对照"**，不是"视差有效率"（后者只证明 SGBM 给了个数，不证明准）。
    按几何残差分档，5–8 m 档上**跨经过 51.1 cm vs 短跨度 46.2 cm** ⇒ 几乎相同
    ⇒ 该量级残差由 `δz = z²/(f·b)·δd` 决定（f·b=25.515、δd=0.3 px）。
  * 但 `<1m` 占比差很多（**94% → 63%**）⇒ 跨经过多出约 1/3 大残差外点 = **误匹配**。
  * 角点为什么在远处：ORB 角点深度中位 **8.51 m**、随机像素 **4.27 m**；
    `z≤5m` 占比 12% vs 45% ⇒ 近场提不出角点。
  * 反证（排除"改参数能救"）：`numDisparities` 64→256 使 `z≤5m` 12%→7%（更差）；
    只取 `z≤4m` 掩码成功率 8%→2%（更差）；换 LK/ORB/描述子同量级。
  * ⚠️ **但内点容差本身也是错源**：第一版 `tol=0.12 m` 比 8.5 m 处噪声（0.85 m）紧 7 倍，
    容差扫描显示成功率 **5/20（0.10 m）→ 16/20（0.80 m）**。
    ⇒ "成功率 2–8%"这个数字**被容差压低**，不能全归给数据。
  * ⚠️ `z≤3 m` 的跨经过**匹配数为 0** ⇒ "近场跨经过准不准"本次数据**答不了**。

短跨度自检（连续关键帧）**是过的**：
  |视觉−DR| 中位 **0.102 m**、比值中位 **0.99**（对比当年稠密 ICP 的 1.88 m）。
  ⚠️ 该自检前提"DR <10 cm 可当真值"未实测，不是精度真值。
  近场（z≤3 m）几何残差 **5.8–7.3 cm**，与 `δz` 公式吻合 ⇒ 估计器在可用深度上是准的。

    .venv/Scripts/python.exe research/tools/xpass_dense.py 20261010_033237
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
REC_ROOT = ROOT / "navmesh_recordings"


def load_time(rec: Path) -> dict[int, float]:
    ev = [json.loads(x) for x in (rec / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return {int(e["k"]): float(e["t"]) for e in ev if e.get("kind") == "kf"}


def project_uv(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """pts（左目光学系 [z,-x,-y]）→ 左目像素 (u,v)。"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = np.maximum(pts[:, 0], 1e-6)
    return np.column_stack([cx - pts[:, 1] * fx / z, cy - pts[:, 2] * fy / z])


class KF:
    """一帧：像素坐标 → base 系坐标的查找（点来自 pts，本身带空洞）。"""

    def __init__(self, rec: Path, k: int, eye_y_from_meta: float | None = None):
        z = np.load(rec / "kf" / f"{k:06d}.npz")
        self.k = k
        self.pts = z["pts"].astype(np.float64)
        self.K = z["K"]
        self.T = np.asarray(z["T_dr"], float)
        self.uv = project_uv(self.pts, self.K)
        self.eye_y = float(z["eye_y"]) if "eye_y" in z else (eye_y_from_meta or 0.0)
        self.tree = cKDTree(self.uv)

    def base_at(self, uv: np.ndarray, r: float = 1.6):
        """uv 邻域内最近的立体点 → 头部 base 系（+eye_y），落空给 NaN。"""
        d, i = self.tree.query(uv, k=1)
        out = self.pts[i].copy()
        out[:, 1] += self.eye_y
        out[d > r] = np.nan
        return out


def tracked_points(kfa: KF, kfb: KF, La, Lb, n_feat=3000, fb_tol=1.0, min_dist=5, r=1.6):
    """**先取有效深度、再取角点**：只在 A 有深度的地方取 Shi-Tomasi 角点，
    再用 LK 追到 B，并要求 B 处也有深度。返回 (base_A, base_B)。"""
    mask = np.zeros(La.shape, np.uint8)
    ui = np.clip(np.rint(kfa.uv[:, 0]).astype(int), 0, La.shape[1] - 1)
    vi = np.clip(np.rint(kfa.uv[:, 1]).astype(int), 0, La.shape[0] - 1)
    mask[vi, ui] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))          # 容忍 1px 误差
    pts = cv2.goodFeaturesToTrack(La, maxCorners=n_feat, qualityLevel=0.01,
                                  minDistance=min_dist, blockSize=7, mask=mask)
    if pts is None or len(pts) < 10:
        return np.zeros((0, 3)), np.zeros((0, 3))
    p1 = pts.reshape(-1, 2).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    p2, st, _ = cv2.calcOpticalFlowPyrLK(La, Lb, p1, None, winSize=(21, 21), maxLevel=3, criteria=crit)
    if p2 is None:
        return np.zeros((0, 3)), np.zeros((0, 3))
    p1b, st2, _ = cv2.calcOpticalFlowPyrLK(Lb, La, p2, None, winSize=(21, 21), maxLevel=3, criteria=crit)
    if p1b is None:
        return np.zeros((0, 3)), np.zeros((0, 3))
    ok = (st.ravel() == 1) & (st2.ravel() == 1)
    ok &= np.linalg.norm(p1b - p1, axis=1) < fb_tol
    p1, p2 = p1[ok], p2[ok]
    if len(p1) < 10:
        return np.zeros((0, 3)), np.zeros((0, 3))
    A = kfa.base_at(p1, r)
    B = kfb.base_at(p2, r)
    good = ~(np.isnan(A[:, 0]) | np.isnan(B[:, 0]))
    return A[good], B[good]


def est_translation(Xa, Xb, Ra, Rb, rng, max_corr=1.5, iters=600, tol=0.12):
    """R 已知（HMD）⇒ 同一物理点满足 R_a·x_a + t_a = R_b·x_b + t_b ⇒ t_b − t_a = R_a·x_a − R_b·x_b。
    单点假设 + 一致性内点（不用残差最小化 ⇒ 对长直墙退化免疫）。返回 (t, 内点数, 内点率)。"""
    n = len(Xa)
    if n < 10:
        return None
    D = Xa @ Ra[:3, :3].T - Xb @ Rb[:3, :3].T
    hor = np.hypot(D[:, 0], D[:, 1])
    if (hor <= max_corr).sum() < 10:
        return None
    Dv = D[hor <= max_corr]
    idx = rng.integers(0, len(Dv), size=min(iters, len(Dv) * 6))
    best, best_in = 0, None
    for i in idx:
        inl = np.linalg.norm(D - Dv[i], axis=1) < tol
        c = int(inl.sum())
        if c > best:
            best, best_in = c, inl
    if best_in is None or best < 10:
        return None
    t = np.median(D[best_in], axis=0)
    inl = np.linalg.norm(D - t, axis=1) < tol
    return t, int(inl.sum()), float(inl.sum()) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--mode", choices=("span", "xpass"), default="span",
                    help="span=短跨度自检（DR 可信）；xpass=跨经过")
    ap.add_argument("--gap", type=float, default=10.0)
    ap.add_argument("--dist", type=float, default=2.0)
    ap.add_argument("--yaw", type=float, default=30.0)
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rec = REC_ROOT / args.rec
    if not (rec / "kf_img").is_dir():
        print(f"[{args.rec}] 没有 kf_img/，无法做原图稠密匹配")
        return
    kt = load_time(rec)
    ids = sorted(k for k in kt if (rec / "kf" / f"{k:06d}.npz").exists())
    rng = np.random.default_rng(args.seed)

    def img(k):
        return cv2.imread(str(rec / "kf_img" / f"{k:06d}_L.png"), cv2.IMREAD_GRAYSCALE)

    if args.mode == "span":
        # 短跨度：连续关键帧，DR 可当真值 —— 这是稠密 ICP 当年失败的对照
        sel = [(ids[a], ids[a + 1]) for a in range(0, min(len(ids) - 1, 200), 5)]
        print(f"[{args.rec}] 短跨度自检 {len(sel)} 对（连续关键帧）")
        dr, vis, diffs = [], [], []
        cache = {}
        for ka, kb in sel:
            if ka not in cache:
                cache[ka] = KF(rec, ka)
            if kb not in cache:
                cache[kb] = KF(rec, kb)
            A, B = tracked_points(cache[ka], cache[kb], img(ka), img(kb))
            if len(A) < 10:
                continue
            r = est_translation(A, B, cache[ka].T, cache[kb].T, rng)
            if r is None:
                continue
            t, ni, fr = r
            d = float(np.linalg.norm(cache[kb].T[:2, 3] - cache[ka].T[:2, 3]))
            dr.append(d)
            vis.append(float(np.hypot(t[0], t[1])))
            diffs.append(abs(vis[-1] - d))
        if dr:
            dr, vis, diffs = np.array(dr), np.array(vis), np.array(diffs)
            print(f"  成功 {len(dr)} 对")
            print(f"  DR 位移 中位 {np.median(dr):.3f} m；视觉 中位 {np.median(vis):.3f} m；比值中位 {np.median(vis/np.maximum(dr,1e-9)):.2f}")
            print(f"  |视觉−DR| 中位 {np.median(diffs):.3f} m  p90 {np.percentile(diffs,90):.3f}")
        out = {"rec": args.rec, "mode": "span", "n": len(dr),
               "dr_med": float(np.median(dr)) if len(dr) else None,
               "vis_med": float(np.median(vis)) if len(vis) else None,
               "absdiff_med": float(np.median(diffs)) if len(diffs) else None}
        print("  ->", json.dumps(out, ensure_ascii=False))
        if args.json:
            Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        return

    # ---- 跨经过 ----
    Z = {k: KF(rec, k) for k in ids}
    P = {k: Z[k].T for k in ids}
    lp = rec / "final_poses.npz"
    LP = {}
    if lp.exists():
        d = np.load(lp)
        LP = {int(i): np.asarray(T, float) for i, T in zip(d["ids"], d["T"])}
    X = np.array([P[k][:2, 3] for k in ids])
    Y = np.array([np.arctan2(P[k][1, 0], P[k][0, 0]) for k in ids])
    T = np.array([kt[k] for k in ids])
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if T[j] - T[i] < args.gap or np.hypot(*(X[j] - X[i])) > args.dist:
                continue
            if abs((Y[j] - Y[i] + np.pi) % (2 * np.pi) - np.pi) > np.radians(args.yaw):
                continue
            pairs.append((i, j))
    rng.shuffle(pairs)
    pairs = pairs[:args.pairs]
    print(f"[{args.rec}] 跨经过候选 {len(pairs)}（Δt≥{args.gap:.0f}s, <{args.dist}m, yaw<{args.yaw:.0f}°）")

    n_ok, nin, nfr, diffs, npt = 0, [], [], [], []
    for i, j in pairs:
        ka, kb = ids[i], ids[j]
        A, B = tracked_points(Z[ka], Z[kb], img(ka), img(kb))
        npt.append(len(A))
        if len(A) < 10:
            continue
        r = est_translation(A, B, P[ka], P[kb], rng)
        if r is None:
            continue
        t, ni, fr = r
        n_ok += 1
        nin.append(ni)
        nfr.append(fr)
        if ka in LP and kb in LP:
            tl = LP[kb][:3, 3] - LP[ka][:3, 3]
            tloop = LP[ka][:3, :3].T @ tl
            diffs.append(float(np.linalg.norm(t - tloop)))
    rate = 100 * n_ok / max(len(pairs), 1)
    print(f"  给出可信平移 {n_ok}（{rate:.0f}%）")
    if npt:
        print(f"  可跟踪点数 中位 {np.median(npt):.0f}")
    if nin:
        print(f"  内点 中位 {np.median(nin):.0f}；内点率 中位 {np.median(nfr)*100:.0f}%")
    if diffs:
        a = np.array(diffs)
        print(f"  |视觉−loop| 帧间位移差 中位 {np.median(a)*100:.1f} cm p90 {np.percentile(a,90)*100:.1f} cm")
    out = {"rec": args.rec, "mode": "xpass", "pairs": len(pairs), "ok": n_ok, "rate": round(rate, 1),
           "inliers_med": float(np.median(nin)) if nin else 0.0,
           "diff_med_cm": float(np.median(diffs) * 100) if diffs else None}
    print("  ->", json.dumps(out, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
