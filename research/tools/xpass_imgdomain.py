# -*- coding: utf-8 -*-
r"""图像域约束能不能救跨经过 —— **判否**（2026-10-10）。

背景：`Docs/点云拖影修复原型与栅格影响（2026-10-09）.md` §11.4 把"图像域约束"
列为出路①，理由是它**绕开"角点必须有可用深度"**这一环，是唯一不撞深度精度预算的方向。

本探针**直接测这条路本身**，结论是**判否**，并且给出判否的完整证据链。
每条都配了"已知答案的对照"，避免再一次用无效判据下结论（那是本仓库复发过的错）。

证据链（--mode 一一对应）
------------------------
1. `control`  **先证明我的机器是好的**：同一帧 L/R 立体对，真值运动已知且精确
   （R = I、t = 基线沿光学 x 轴）。我的管线在这上面给出 **R 误差 0.00°、t 方向 0.00°**。
   ⇒ 后面所有数字都不是"我写错了坐标系"造成的。
2. `synth`    合成已知 R/t（深度 3.6 m、基线 0.70 m、旋转 17.4°、0.5 px 噪声）
   ⇒ 平移方向误差 **3.4°**。⇒ 估计器在**真实参数下**是有能力的。
3. `synth` 内注入错配：即使 **80% 完全随机错配**，方向误差也只 10°。
   ⇒ "跨经过失败"**不能**归因于误匹配多。
4. `truth`    用真值运动 E_gt 直接量匹配：真值内点率（Sampson<1.5 px）
   * 短跨度 **10–15%**、跨经过 **0.0%**
   而 `findEssentialMat` 自报内点率 **86–88%** ⇒ **假共识**：
   RANSAC 锁到了一个自洽但错误的子集。
   ⚠️ 这正是"视差有效率 93%"那类无效判据的复发形态 ——
   "估计器返回了东西"≠"返回的东西对"。
5. `plane`    用真值位姿 + **枚举深度平面与符号**，取最佳 NCC：
   短跨度 **0.68**、跨经过 **0.21**、L/R 对照 **0.94**。
   ⇒ 位姿驱动的图像对齐也不行。
6. `poseless` **完全不用位姿**（phaseCorrelate / ECC 欧氏·仿射·单应 / 稠密光流）：
   跨经过 ECC 仿射 **0.31**、短跨度 **0.86**、L/R **0.98**。
   ⇒ 连"让图像自己对"都上不去 ⇒ 不是估计器问题。
7. `band`     把 **loop 位姿距离**当自变量：相似度**单调下降**
   （NCC 0.195 @≤0.6 m → 0.001 @12–30 m）⇒ **配对是真的**，不是把不同地方拉到一起。
8. `band` 的**决定性对照**：跨经过（同地、≤0.6 m）ECC 0.28
   vs **远间隔（8–20 m，必然不同地）ECC 0.24** ⇒ **两者分不开**。
   而连续关键帧 0.87。分块口径同样：能对齐的块占比 跨经过 **18%** vs 远间隔 **17%**、连续帧 **73%**。
9. `photometric` 分离成因：直方图均衡把跨经过从 0.40 抬到 0.52、远间隔 0.19→0.29
   ⇒ 提升有限且两者仍按同比例平移 ⇒ **不是曝光/白平衡能解释的**。

关键对照（同一批数据、同一个位移量级）
------------------------------------
  位移中位 **0.69 m**（两者相同！），只有**间隔时间**不同：
    间隔   4 s  → ECC 0.85
    间隔 216 s  → ECC 0.33
  ⇒ 图像相似度**随时间显著退化**。

⚠️ **但"时间"不是唯一因素 —— 两个因素都在起作用**（两因素实测）：
  * 固定 loop 距离 ≤0.6 m，只变 dt：0.53（10–30 s）→ 0.36（30–60 s）→ 0.33（120–300 s）
  * 固定 dt 10–30 s，只变 loop 距离：0.53（≤0.6 m）→ 0.45（0.6–2 m）→ 0.27（2–6 m）→ 0.17（6–20 m）
  ⇒ **位置**与**时间**都对相似度有独立影响。

判定（**不是"图像里没有信号"，而是"信号太弱、且现有方法取不出来"**）
----------------------------------------------------------------
必须把两件事分开说，否则又是一次过度归因：

* **信号确实存在**：同一 dt 下"同地"（0.53）明显高于"异地"（0.17），约 **2–3×**；
  `band` 的分块口径在 n=10…60 上稳定保持 **≈26% vs ≈13%**（跨经过 / 远间隔）。
* **但强度远不够**：跨经过最优 0.34–0.52，而连续关键帧 **0.85**、L/R **0.98**；
  真值内点率 **0.0%**（短跨度 12%），且 `findEssentialMat` 在 0.0% 真值下自报 **87.5%**
  ⇒ **假共识**。三种无位姿方法（相位相关 / ECC / 稠密光流）全部在 0.34 触顶。

⇒ **§11.4 出路①（图像域约束）判否**：不是"估计器不够强"、不是"参数没调好"、
不是"坐标系写错"（`control` 0.00° 已排除），而是**跨经过的图像信号强度不足以
支撑一个可用平移估计**。

⇒ 跨经过对齐这条线：稀疏 ORB（§9.1）判否、子图平移相关（§9.2）判否、
  带原图稠密对齐（§11）不用深度也不行、图像域约束（本文）也不行
  ⇒ **建议停止在这条线上继续投入**，回到 `repl,10,2`
  （唯一有在线因果验证、但有场次依赖的做法）。

未来若要重开这条线，前提是**先把信号强度提上去**（更短的重访间隔、
或掩掉动态区域后重测），而不是再换一个估计器 —— 换估计器已被证否。

复现
----
    .venv/Scripts/python.exe research/tools/xpass_imgdomain.py 20261010_033237 --mode control
    .venv/Scripts/python.exe research/tools/xpass_imgdomain.py 20261010_033237 --mode synth
    .venv/Scripts/python.exe research/tools/xpass_imgdomain.py 20261010_033237 --mode truth
    .venv/Scripts/python.exe research/tools/xpass_imgdomain.py 20261010_033237 --mode plane
    .venv/Scripts/python.exe research/tools/xpass_imgdomain.py 20261010_033237 --mode poseless
    .venv/Scripts/python.exe research/tools/xpass_imgdomain.py 20261010_033237 --mode band
    .venv/Scripts/python.exe research/tools/xpass_imgdomain.py 20261010_033237 --mode photometric
    .venv/Scripts/python.exe research/tools/xpass_imgdomain.py 20261010_033237 --mode twofactor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REC_ROOT = ROOT / "navmesh_recordings"
# 光学系（x 右 y 下 z 前）→ base（x 前 y 左 z 上），与 nav_loop.M_OPT 相同
M_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
BASELINE_M = 0.126
MIN_GAP_S = 10.0
SAMPSON_TOL = 1.5          # 与 findEssentialMat 的内点门一致


# ---------------------------------------------------------------- 基础
def load_time(rec: Path) -> dict[int, float]:
    ev = [json.loads(x) for x in
          (rec / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    return {int(e["k"]): float(e["t"]) for e in ev if e.get("kind") == "kf"}


def load_final_poses(rec: Path) -> dict[int, np.ndarray]:
    d = np.load(rec / "final_poses.npz")
    return {int(i): np.asarray(T, float) for i, T in zip(d["ids"], d["T"])}


def load_gray(rec: Path, k: int, eye: str = "L"):
    im = cv2.imread(str(rec / "kf_img" / f"{k:06d}_{eye}.png"), cv2.IMREAD_GRAYSCALE)
    return None if im is None else im


def sampson(E: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    x1 = np.column_stack([p1, np.ones(len(p1))])
    x2 = np.column_stack([p2, np.ones(len(p2))])
    Ex1 = (E @ x1.T).T
    Etx2 = (E.T @ x2.T).T
    num = np.abs(np.sum(x2 * Ex1, axis=1))
    den = np.sqrt(Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2)
    return num / np.maximum(den, 1e-12)


def E_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], float)
    return tx @ R


def truth_opt(Ta: np.ndarray, Tb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """map→cam 位姿对 → 左目光学系相对运动 (R, t)，X_b = R X_a + t。

    X_b = R_b^T R_a X_a + R_b^T (t_a − t_b) = R_rel X_a + t_rel（base 系）
    再换到光学系：R_opt = M R_rel M^T、t_opt = M t_rel（M 正交，共轭是 M·R·M^T）。
    ⚠️ 写成 M^T R M 会让三角化为正的比例从 91.5% 塌到 ~12%（实测），是本仓库踩过的坑。
    """
    Rrel = Tb[:3, :3].T @ Ta[:3, :3]
    trel = Tb[:3, :3].T @ (Ta[:3, 3] - Tb[:3, 3])
    return M_OPT @ Rrel @ M_OPT.T, M_OPT @ trel


def rot_err_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = (np.trace(np.asarray(R1).T @ np.asarray(R2)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def ang_deg(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel()[:3], np.asarray(b, float).ravel()[:3]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(float(a @ b) / (na * nb), -1.0, 1.0))))


def mutual(ga, gb, orb, bf, ratio: float = 0.75, nmin: int = 12):
    """ORB 互相最近邻（比例检验 + 双向一致性）。返回 (p1, p2) 或 None。"""
    kpa, da = orb.detectAndCompute(ga, None)
    kpb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(kpa) < 8 or len(kpb) < 8:
        return None
    fwd = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(da, db, k=2)
           if len(m) == 2 and m[0].distance < ratio * m[1].distance}
    rev = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(db, da, k=2)
           if len(m) == 2 and m[0].distance < ratio * m[1].distance}
    mm = [(q, t) for q, t in fwd.items() if rev.get(t) == q]
    if len(mm) < nmin:
        return None
    return (np.array([kpa[q].pt for q, _ in mm], np.float32),
            np.array([kpb[t].pt for _, t in mm], np.float32))


def med(v) -> float:
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(v)) if v else float("nan")


def ncc(a, b, mask=None) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if mask is not None:
        m = mask > 0
        if m.sum() < 500:
            return float("nan")
        a, b = a[m], b[m]
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-9 else float("nan")


# ---------------------------------------------------------------- 选对
def pairs_span(ids: list[int], n: int, step: int = 7) -> list[tuple[int, int]]:
    """短跨度：连续关键帧（位移 ≈ kf_dist_m，几何良好）。"""
    return [(ids[a], ids[a + 1]) for a in range(0, min(len(ids) - 1, 500), step)][:n]


def pairs_xpass(ids, kt, FP, n, max_dist: float = 0.6, max_yaw: float = 25.0,
                seed: int = 0) -> list[tuple[int, int]]:
    """跨经过：**必须用 loop/final 位姿筛**。

    ⚠️ 绝不能用 dr 位姿筛 —— FACTS §六 已登记：DR 本身飘 2.3 m，
    按它筛出的"<1 m 同地"对在 loop 位姿下其实相距 2–6 m（实测 8 对全是伪对）。
    """
    P = {k: FP[k][:2, 3] for k in ids}
    Y = {k: np.arctan2(FP[k][1, 0], FP[k][0, 0]) for k in ids}
    out = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if kt[b] - kt[a] < MIN_GAP_S:
                continue
            if np.hypot(*(P[b] - P[a])) > max_dist:
                continue
            if abs((Y[b] - Y[a] + np.pi) % (2 * np.pi) - np.pi) > np.radians(max_yaw):
                continue
            out.append((a, b))
    rng = np.random.default_rng(seed)
    rng.shuffle(out)
    return out[:n]


def pairs_far(ids, kt, FP, n, dlo: float = 8.0, dhi: float = 20.0, seed: int = 0):
    """远间隔对照：必然不同地（用来确认"预处理/方法不会把不可能对齐的也抬高"）。"""
    P = {k: FP[k][:2, 3] for k in ids}
    out = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if kt[b] - kt[a] < MIN_GAP_S:
                continue
            if not (dlo <= np.hypot(*(P[b] - P[a])) <= dhi):
                continue
            out.append((a, b))
    rng = np.random.default_rng(seed)
    rng.shuffle(out)
    return out[:n]


# ---------------------------------------------------------------- 模式
def mode_control(rec, ids, kt, FP, n, args) -> dict:
    """同一帧 L/R 立体对：真值 R = I、t = [基线, 0, 0]（左目光学系）。"""
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    errR, angX, vecs, nok = [], [], [], 0
    sel = ids[:: max(1, len(ids) // n)][:n]
    for k in sel:
        ga, gb = load_gray(rec, k, "L"), load_gray(rec, k, "R")
        if ga is None or gb is None:
            continue
        K = np.load(rec / "kf" / f"{k:06d}.npz")["K"]
        r = mutual(ga, gb, orb, bf)
        if r is None:
            continue
        E, mask = cv2.findEssentialMat(r[0], r[1], K, cv2.RANSAC, 0.999, SAMPSON_TOL)
        if E is None or E.shape[0] < 3:
            continue
        _, R, t, m = cv2.recoverPose(E, r[0], r[1], K, mask=mask.copy())
        if m is None or int(m.sum()) < 15:
            continue
        nok += 1
        errR.append(rot_err_deg(np.eye(3), R))
        tv = t.ravel()[:3]
        angX.append(min(ang_deg(tv, [1, 0, 0]), ang_deg(tv, [-1, 0, 0])))
        vecs.append(tv / max(np.linalg.norm(tv), 1e-9))
    out = {"n_ok": nok, "n_try": len(sel),
           "rot_err_deg": med(errR), "t_axis_err_deg": med(angX)}
    if vecs:
        vm = np.median(np.array(vecs), axis=0)
        out["t_dir_median"] = [round(float(x), 3) for x in vm]
    return out


def _synth_trial(depth, base, rot_deg, noise_px, npts, K, seed, out_frac=0.0):
    rng = np.random.default_rng(seed)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = depth * (1.0 + 0.25 * rng.standard_normal(npts))
    X = rng.uniform(-0.6, 0.6, npts) * Z
    Y = rng.uniform(-0.35, 0.35, npts) * Z
    P = np.column_stack([X, Y, Z])
    th = np.radians(rot_deg)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    t = np.array([base, 0.0, 0.0])
    P2 = P @ R.T + t
    if (P2[:, 2] <= 0.1).any():
        return None

    def proj(Q):
        return np.column_stack([fx * Q[:, 0] / Q[:, 2] + cx, fy * Q[:, 1] / Q[:, 2] + cy])
    p1 = proj(P) + rng.normal(0, noise_px, (npts, 2))
    p2 = proj(P2) + rng.normal(0, noise_px, (npts, 2))
    nout = int(round(out_frac * npts))
    if nout > 0:                      # 注入**完全随机**错配（等价"匹配器彻底失效"上限）
        idx = rng.choice(npts, nout, replace=False)
        p2[idx] = np.column_stack([rng.uniform(0, 720, nout), rng.uniform(0, 405, nout)])
    p1 = p1.astype(np.float32)
    p2 = p2.astype(np.float32)
    E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, SAMPSON_TOL)
    if E is None or E.shape[0] < 3:
        return None
    _, Rr, tr, m = cv2.recoverPose(E, p1, p2, K, mask=mask.copy())
    if m is None or int(m.sum()) < 10:
        return None
    return {"rot": rot_err_deg(R, Rr), "dir": ang_deg(M_OPT @ t, M_OPT @ tr.ravel()[:3]),
            "inl_frac": float(m.mean())}


def mode_synth(rec, ids, kt, FP, n, args) -> dict:
    """合成已知 R/t：证明"估计器在真实参数下有能力"，并给错配注入的上限。"""
    K = np.array([[202.5, 0, 360.0], [0, 202.5, 202.5], [0, 0, 1.0]])
    cases = [
        ("真实短跨度参数", 3.6, 0.70, 1.9, 0.5, 300),
        ("真实跨经过参数", 3.6, 0.70, 17.4, 0.5, 300),
        ("无噪声（上界）", 3.6, 0.70, 17.4, 0.0, 300),
        ("纯平移", 3.6, 0.70, 0.0, 0.5, 300),
        ("远点 8 m", 8.0, 0.70, 1.9, 0.5, 300),
        ("大基线 2 m", 3.6, 2.00, 1.9, 0.5, 300),
    ]
    out = {"cases": []}
    for name, depth, base, rot, noise, npts in cases:
        rs = [r for r in (_synth_trial(depth, base, rot, noise, npts, K, s) for s in range(12)) if r]
        out["cases"].append({"name": name, "depth": depth, "base": base, "rot": rot,
                             "noise_px": noise,
                             "rot_err_deg": med([r["rot"] for r in rs]),
                             "dir_err_deg": med([r["dir"] for r in rs]),
                             "inl_frac": med([r["inl_frac"] for r in rs])})
    out["outlier_injection"] = []
    for frac in (0.0, 0.4, 0.6, 0.8):
        rs = [r for r in (_synth_trial(3.6, 0.70, 17.4, 0.5, 300, K, s, frac)
                          for s in range(16)) if r]
        out["outlier_injection"].append({"outlier_frac": frac,
                                         "dir_err_deg": med([r["dir"] for r in rs])})
    return out


def mode_truth(rec, ids, kt, FP, n, args) -> dict:
    """真值内点率 vs RANSAC 自报内点率 —— 揭穿**假共识**。"""
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def run(pairs, label):
        rows = []
        for ka, kb in pairs:
            ga, gb = load_gray(rec, ka), load_gray(rec, kb)
            if ga is None or gb is None:
                continue
            K = np.load(rec / "kf" / f"{ka:06d}.npz")["K"]
            r = mutual(ga, gb, orb, bf)
            if r is None:
                continue
            R, t = truth_opt(FP[ka], FP[kb])
            s_gt = sampson(E_from_Rt(R, t), r[0], r[1])
            E, mask = cv2.findEssentialMat(r[0], r[1], K, cv2.RANSAC, 0.999, SAMPSON_TOL)
            ric = float("nan")
            if E is not None and E.shape[0] >= 3:
                _, _, _, m = cv2.recoverPose(E, r[0], r[1], K, mask=mask.copy())
                if m is not None:
                    ric = float(m.mean())
            rows.append({"n": len(r[0]), "truth_in": float((s_gt < SAMPSON_TOL).mean()),
                         "s_gt_px": float(np.median(s_gt)), "ransac_in": ric})
        return {"label": label, "n_pairs": len(rows),
                "matches": med([r["n"] for r in rows]),
                "truth_inlier_frac": med([r["truth_in"] for r in rows]),
                "ransac_inlier_frac": med([r["ransac_in"] for r in rows]),
                "truth_sampson_px": med([r["s_gt_px"] for r in rows])}

    return {"span": run(pairs_span(ids, n), "短跨度"),
            "xpass": run(pairs_xpass(ids, kt, FP, n), "跨经过")}


def mode_plane(rec, ids, kt, FP, n, args) -> dict:
    """真值位姿 + 枚举深度平面与符号，取最佳 NCC（位姿驱动的对齐上限）。"""
    depths = [0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0, 15.0, 25.0]

    def best_plane(ga, gb, K, R, t):
        h, w = ga.shape
        out = []
        for d in depths:
            for sgn in (+1, -1):
                Hm = K @ (R + sgn * np.outer(t, np.array([0.0, 0.0, 1.0])) / d) @ np.linalg.inv(K)
                try:
                    gw = cv2.warpPerspective(gb, Hm, (w, h), borderMode=cv2.BORDER_CONSTANT)
                    mk = cv2.warpPerspective(np.full_like(gb, 255), Hm, (w, h),
                                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                except cv2.error:
                    continue
                v = ncc(ga, gw, mk)
                if not np.isnan(v):
                    out.append((v, d))
        return max(out) if out else (float("nan"), float("nan"))

    def run(pairs, label):
        best, base = [], []
        for ka, kb in pairs:
            ga, gb = load_gray(rec, ka), load_gray(rec, kb)
            if ga is None or gb is None:
                continue
            K = np.load(rec / "kf" / f"{ka:06d}.npz")["K"]
            R, t = truth_opt(FP[ka], FP[kb])
            v, d = best_plane(ga, gb, K, R, t)
            best.append(v)
            base.append(ncc(ga, gb))
        return {"label": label, "n_pairs": len(best), "base_ncc": med(base),
                "best_ncc": med(best), "best_ncc_p90": float(np.percentile(best, 90)) if best else None}

    ctrl = []
    for k in ids[:: max(1, len(ids) // max(n, 1))][:n]:
        ga, gb = load_gray(rec, k, "L"), load_gray(rec, k, "R")
        if ga is None or gb is None:
            continue
        K = np.load(rec / "kf" / f"{k:06d}.npz")["K"]
        ctrl.append(best_plane(ga, gb, K, np.eye(3), np.array([BASELINE_M, 0.0, 0.0]))[0])
    return {"LR_control": {"n": len(ctrl), "best_ncc": med(ctrl)},
            "span": run(pairs_span(ids, n), "短跨度"),
            "xpass": run(pairs_xpass(ids, kt, FP, n), "跨经过")}


def mode_poseless(rec, ids, kt, FP, n, args) -> dict:
    """**完全不用位姿**的图像域对齐（原提案那一类方法本身）。"""
    def ecc(ga, gb, mode, iters=150):
        wa = np.eye(2, 3, dtype=np.float32)
        try:
            cc, _ = cv2.findTransformECC(ga, gb, wa, mode,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6),
                                         None, 5)
            return float(cc)
        except cv2.error:
            return float("nan")

    def flow_fb(ga, gb):
        a = (ga * 255).astype(np.uint8)
        b = (gb * 255).astype(np.uint8)
        fa = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        fb = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        h, w = a.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xb = np.clip(xx + fa[..., 0], 0, w - 1)
        yb = np.clip(yy + fa[..., 1], 0, h - 1)
        fbx = cv2.remap(fb[..., 0], xb, yb, cv2.INTER_LINEAR)
        fby = cv2.remap(fb[..., 1], xb, yb, cv2.INTER_LINEAR)
        err = np.hypot(fa[..., 0] + fbx, fa[..., 1] + fby)
        return float((err < 1.0).mean())

    def one(ga, gb):
        f = ga.astype(np.float32) / 255.0
        g = gb.astype(np.float32) / 255.0
        (dx, dy), resp = cv2.phaseCorrelate(f, g)
        return {"phase": float(resp), "phase_shift_px": float(np.hypot(dx, dy)),
                "ecc_euclidean": ecc(f, g, cv2.MOTION_EUCLIDEAN),
                "ecc_affine": ecc(f, g, cv2.MOTION_AFFINE),
                "flow_fb_frac": flow_fb(ga, gb)}

    def run(pairs, label):
        rows = []
        for ka, kb in pairs:
            ga, gb = load_gray(rec, ka), load_gray(rec, kb)
            if ga is None or gb is None:
                continue
            rows.append(one(ga, gb))
        return {"label": label, "n_pairs": len(rows),
                **{k: med([r[k] for r in rows]) for k in
                   ("phase", "phase_shift_px", "ecc_euclidean", "ecc_affine", "flow_fb_frac")}}

    lr = []
    for k in ids[:: max(1, len(ids) // max(n, 1))][:n]:
        ga, gb = load_gray(rec, k, "L"), load_gray(rec, k, "R")
        if ga is not None and gb is not None:
            lr.append(one(ga, gb))
    return {"LR_control": {"n": len(lr),
                           **{k: med([r[k] for r in lr]) for k in ("phase", "ecc_euclidean", "ecc_affine")}},
            "span": run(pairs_span(ids, n), "短跨度"),
            "xpass": run(pairs_xpass(ids, kt, FP, n), "跨经过")}


def mode_band(rec, ids, kt, FP, n, args) -> dict:
    """把 loop 距离当自变量 + **远间隔对照**（决定性：同地 vs 异地在图像上可辨吗）。"""
    def ecc_affine(ga, gb):
        wa = np.eye(2, 3, dtype=np.float32)
        try:
            cc, _ = cv2.findTransformECC(ga.astype(np.float32) / 255.0, gb.astype(np.float32) / 255.0,
                                         wa, cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
                                         None, 5)
            return float(cc)
        except cv2.error:
            return float("nan")

    def tile_frac(ga, gb, rows=6, cols=8, thr=0.6):
        h, w = ga.shape
        th, tw = h // rows, w // cols
        ccs = []
        for r in range(rows):
            for c in range(cols):
                ta = ga[r * th:(r + 1) * th, c * tw:(c + 1) * tw].astype(np.float32) / 255.0
                tb = gb[r * th:(r + 1) * th, c * tw:(c + 1) * tw].astype(np.float32) / 255.0
                if ta.std() < 1e-3 or tb.std() < 1e-3:
                    continue
                wa = np.eye(2, 3, dtype=np.float32)
                try:
                    cc, _ = cv2.findTransformECC(ta, tb, wa, cv2.MOTION_TRANSLATION,
                                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
                                                 None, 3)
                    if not np.isnan(cc):
                        ccs.append(float(cc))
                except cv2.error:
                    continue
        return float((np.array(ccs) > thr).mean()) if ccs else float("nan")

    P = {k: FP[k][:2, 3] for k in ids}
    Y = {k: np.arctan2(FP[k][1, 0], FP[k][0, 0]) for k in ids}

    def sample(dlo, dhi, cnt):
        got = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if kt[b] - kt[a] < MIN_GAP_S:
                    continue
                dd = float(np.hypot(*(P[b] - P[a])))
                if not (dlo <= dd < dhi):
                    continue
                if abs((Y[b] - Y[a] + np.pi) % (2 * np.pi) - np.pi) > np.radians(25.0):
                    continue
                got.append((a, b))
        rng = np.random.default_rng(0)
        rng.shuffle(got)
        return got[:cnt]

    bands = []
    for dlo, dhi in ((0.0, 0.6), (0.6, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 30.0)):
        s = sample(dlo, dhi, n)
        ea, nc = [], []
        for a, b in s:
            ga, gb = load_gray(rec, a), load_gray(rec, b)
            if ga is None or gb is None:
                continue
            ea.append(ecc_affine(ga, gb))
            x = ga.astype(np.float32) - float(ga.mean())
            y = gb.astype(np.float32) - float(gb.mean())
            nc.append(float(x.ravel() @ y.ravel() / (np.linalg.norm(x) * np.linalg.norm(y))))
        bands.append({"lo": dlo, "hi": dhi, "n": len(s), "ecc_affine": med(ea), "ncc": med(nc)})

    def tile_report(pairs, label):
        fr, bs = [], []
        for a, b in pairs:
            ga, gb = load_gray(rec, a), load_gray(rec, b)
            if ga is None or gb is None:
                continue
            fr.append(tile_frac(ga, gb))
        return {"label": label, "n_pairs": len(fr), "aligned_tile_frac": med(fr)}

    span = pairs_span(ids, n)
    xp = pairs_xpass(ids, kt, FP, n)
    far = pairs_far(ids, kt, FP, n)
    return {"distance_bands": bands,
            "tiles": {"span": tile_report(span, "短跨度"),
                      "xpass": tile_report(xp, "跨经过"),
                      "far": tile_report(far, "远间隔 8–20 m")},
            "ecc_affine_summary": {
                "span": med([ecc_affine(load_gray(rec, a), load_gray(rec, b))
                             for a, b in span
                             if load_gray(rec, a) is not None and load_gray(rec, b) is not None]),
                "xpass": med([ecc_affine(load_gray(rec, a), load_gray(rec, b))
                              for a, b in xp
                              if load_gray(rec, a) is not None and load_gray(rec, b) is not None]),
                "far": med([ecc_affine(load_gray(rec, a), load_gray(rec, b))
                            for a, b in far
                            if load_gray(rec, a) is not None and load_gray(rec, b) is not None])}}


def mode_photometric(rec, ids, kt, FP, n, args) -> dict:
    """成因分离：光度（可修）vs 内容（不可修）。"""
    def norm_ms(x):
        m, s = float(x.mean()), float(x.std())
        return (x - m) / (s if s > 1e-6 else 1.0)

    def hist_eq(x):
        return cv2.equalizeHist(x.astype(np.uint8)).astype(np.float32)

    def clahe(x):
        return cv2.createCLAHE(2.0, (8, 8)).apply(x.astype(np.uint8)).astype(np.float32)

    def gradmag(x):
        gx = cv2.Sobel(x.astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x.astype(np.uint8), cv2.CV_32F, 0, 1, ksize=3)
        return norm_ms(np.hypot(gx, gy))

    preps = [("原始", lambda x: x.astype(np.float32)), ("均值方差归一", norm_ms),
             ("直方图均衡", hist_eq), ("CLAHE", clahe), ("梯度幅值", gradmag)]

    def ecc_affine(ga, gb):
        wa = np.eye(2, 3, dtype=np.float32)
        try:
            cc, _ = cv2.findTransformECC(ga, gb, wa, cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
                                         None, 5)
            return float(cc)
        except cv2.error:
            return float("nan")

    def run(pairs, label):
        acc = {nm: [] for nm, _ in preps}
        for a, b in pairs:
            ga, gb = load_gray(rec, a), load_gray(rec, b)
            if ga is None or gb is None:
                continue
            for nm, f in preps:
                acc[nm].append(ecc_affine(f(ga), f(gb)))
        return {"label": label, **{nm: med(v) for nm, v in acc.items()}}

    return {"span": run(pairs_span(ids, n), "短跨度"),
            "xpass": run(pairs_xpass(ids, kt, FP, n), "跨经过"),
            "far": run(pairs_far(ids, kt, FP, n), "远间隔 8–20 m")}


def mode_twofactor(rec, ids, kt, FP, n, args) -> dict:
    """两因素分离：**时间**与**位置**各自对图像相似度的独立影响。

    这是"信号存在但太弱"这个判断的依据 —— 只测其中一个因素会得出错误结论：
      * 固定 loop 距离、只变 dt ⇒ 看时间效应
      * 固定 dt、只变 loop 距离 ⇒ 看位置效应
    """
    def ecc_affine(ga, gb):
        wa = np.eye(2, 3, dtype=np.float32)
        try:
            cc, _ = cv2.findTransformECC(ga.astype(np.float32) / 255.0, gb.astype(np.float32) / 255.0,
                                         wa, cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
                                         None, 5)
            return float(cc)
        except cv2.error:
            return float("nan")

    P = {k: FP[k][:2, 3] for k in ids}
    Y = {k: np.arctan2(FP[k][1, 0], FP[k][0, 0]) for k in ids}

    def collect(pred, cnt):
        got = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if pred(a, b):
                    got.append((a, b))
        rng = np.random.default_rng(0)
        rng.shuffle(got)
        return got[:cnt]

    def score(pr):
        v = []
        for a, b in pr:
            ga, gb = load_gray(rec, a), load_gray(rec, b)
            if ga is None or gb is None:
                continue
            v.append(ecc_affine(ga, gb))
        return med(v), len(pr)

    # ---- 因素 1：时间（固定 loop 距离 ≤0.6 m）----
    time_effect = []
    for tlo, thi in ((10, 30), (30, 60), (60, 120), (120, 300)):
        pr = collect(lambda a, b, tlo=tlo, thi=thi: (
            tlo <= kt[b] - kt[a] < thi
            and np.hypot(*(P[b] - P[a])) <= 0.6
            and abs((Y[b] - Y[a] + np.pi) % (2 * np.pi) - np.pi) <= np.radians(25)), n)
        v, cnt = score(pr)
        time_effect.append({"dt_s": [tlo, thi], "n": cnt, "ecc_affine": v})

    # ---- 因素 2：位置（固定 dt 10–30 s）----
    dist_effect = []
    for dlo, dhi in ((0.0, 0.6), (0.6, 2.0), (2.0, 6.0), (6.0, 20.0)):
        pr = collect(lambda a, b, dlo=dlo, dhi=dhi: (
            10 <= kt[b] - kt[a] < 30
            and dlo <= np.hypot(*(P[b] - P[a])) < dhi
            and abs((Y[b] - Y[a] + np.pi) % (2 * np.pi) - np.pi) <= np.radians(25)), n)
        v, cnt = score(pr)
        dist_effect.append({"loop_dist_m": [dlo, dhi], "n": cnt, "ecc_affine": v})

    return {"time_effect_fixed_place": time_effect,
            "place_effect_fixed_time": dist_effect}


MODES = {"control": mode_control, "synth": mode_synth, "truth": mode_truth,
         "plane": mode_plane, "poseless": mode_poseless, "band": mode_band,
         "photometric": mode_photometric, "twofactor": mode_twofactor}


def main() -> None:
    ap = argparse.ArgumentParser(description="图像域约束救不救得了跨经过（判否证据链）")
    ap.add_argument("rec")
    ap.add_argument("--mode", choices=sorted(MODES), default="truth")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    rec = REC_ROOT / args.rec
    if not (rec / "kf_img").is_dir():
        print(f"[{args.rec}] 没有 kf_img/，无法做图像域测试")
        return
    kt = load_time(rec)
    FP = load_final_poses(rec) if (rec / "final_poses.npz").exists() else {}
    if args.mode != "synth" and not FP:
        print(f"[{args.rec}] 缺 final_poses.npz ⇒ 无法用 loop 位姿筛真同地对"
              f"（⚠️ 不要退回用 dr 位姿筛，FACTS §六 已登记那是伪对）")
        return
    ids = sorted(k for k in kt if (rec / "kf_img" / f"{k:06d}_L.png").exists()
                 and (args.mode == "synth" or k in FP))
    if not ids:
        print(f"[{args.rec}] 没有可用的关键帧")
        return
    print(f"[{args.rec}] 模式={args.mode}  kf_img∩位姿 = {len(ids)} 帧")
    out = MODES[args.mode](rec, ids, kt, FP, args.pairs, args)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()