# -*- coding: utf-8 -*-
r"""残余漂移的**形态**诊断：是常量朝向偏移（一行可修）、尺度误差，还是随机游走？

    python research/tools/drift_shape.py 20261001_044153 20260929_045615

为什么需要它
------------
`Docs/archive/停顿后地图错位-根因诊断（2026-10-01）.md` §九 量到「真实漂移是路程的 2.3–4.5 %、
随路程**线性增长**」，但**成因没有查**。已经排除掉两个：

* **尺度**：`loop_selfcal.py` 长基线 OSC/几何 = 1.0019（044153）⇒ 两条链尺度一致；
* **DR 矢量 bug**：已修（`Docs/archive/航位推算矢量位移修正（2026-09-24）.md`：闭环前漂移 5.243→3.688 m）。

剩下三种形态，本工具用**两条互相独立的证据**区分：

证据 A（只用 `events.jsonl`，完全不碰地图）
    ``correction_m`` = 这条回环想让图偏移多少（世界米），``dist_m`` 差 = 两帧间的路程。
    比值 c/L 若不随基线变化 ⇒ 线性漂移；若随基线收敛 ⇒ 尺度。

证据 B（`kf/*.npz` 的 ``T_dr`` vs ``final_poses.npz`` 的 ``T_map``）
    ``r(k) = p_map − p_dr`` 就是回环累积施加的修正场。对 r 拟合一个全局小变换
    ``r ≈ A·p_dr + t``，把 ``A`` 拆成**对称部分（尺度）**与**反对称部分（常量朝向偏移 ω）**：

    * ``A ≈ 纯反对称``（ω ≠ 0 而尺度 ≈ 0）⇒ **整条 DR 轨迹被转了一个常量角**，
      来源是速度-朝向的帧失配或时间滞后 ⇒ 一行可修；
    * ``A`` 有显著对称部分 ⇒ 尺度误差；
    * 两者都小、拟合后残差仍大 ⇒ **非刚体漂移**（随机游走），要改运动模型。

⚠️ 诚实边界
------------
* 证据 B 用的是 ``T_map``（回环修正**之后**的轨迹），所以它量的是「回环认为该修多少」，
  不是独立的真值。它能定性区分上述三种形态，不能反过来证明回环修得对。
* ``T_dr`` / ``T_map`` 都是**追踪米**，本工具统一乘 ``world_scale`` 换成世界米再比较。

🔴 **证据 B 的「尺度」不是 DR 尺度误差的读数**（2026-10-15，`affine_caliber.py`）
--------------------------------------------------------------------------
受控注入实测（只把 ``T_dr`` 乘已知倍率、**回环约束不动**）：注入 **30%** 的已知尺度误差，
证据 B 的尺度只动 **2–6 个百分点** ⇒ **传函增益只有 0.05–0.11**（``=1`` 才配当 DR 尺度读）。

机制：``nav_loop.LoopCloser.optimize`` 里里程边权 ``ow = 1/σ_odo²``（录制值 0.03 m），
回环边 ``1/σ_loop²``（0.1 m）⇒ 每条里程边刚度是回环边的 **~11 倍**，且条数远多于回环边
⇒ **地图尺寸基本跟着 DR 走**（实测 ``map_rms/dr_rms ≈ 1.00``），回环拉不住它。
把 ``σ_odo`` 由 0.03 放到 0.5，增益升到 0.307 ⇒ 机制确认；而同一份数据只换 ``σ_odo``，
``044153`` 的尺度就从 ``+0.78%`` 变到 ``−0.94%`` ⇒ **它读的是位姿图权重**。

**可以用于**：描述「地图与 DR 之间的修正场」有多大、形态判别（反对称 / 对称 / 都小）。
**不得用于**：推断 DR 链的尺度或朝向误差、判断「哪个场次的 DR 更准」、跨场次排名。
要 DR 的独立尺度读数请用 ``stereo_geom_chain.py``（不吃 OSC、也不吃回环）。

⚠️ **口径**：``kf_max_age_s = 3.0`` 让**站着不动也每 3 秒补一个关键帧**，所以关键帧
**不是按位置均匀取的**（``045615`` 实测 **48.1%** 的关键帧 ``T_dr`` 位置重合）。
无权最小二乘会把同一地点放大上千倍。证据 B 因此同时报 **all / loc（按位置去重）/ dwell
（1/驻留数加权）** 三列；重合 >5% 时以 **loc** 为准。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EDGES = [0.0, 1.0, 2.0, 4.0, 8.0, 1e9]


def load(rec: Path, final_npz: Path | None = None
         ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, float],
                    list[dict], float, bool]:
    """返回 (T_dr 平面位置, T_map 平面位置, 每个关键帧的累计路程, 回环列表, world_scale, 是否拿到权威最终位姿)。"""
    meta = json.loads((rec / "meta.json").read_text(encoding="utf-8"))
    ws = float(meta["config"]["mapper"]["world_scale"])

    dr: dict[int, np.ndarray] = {}
    mp: dict[int, np.ndarray] = {}
    dist: dict[int, float] = {}
    for f in (rec / "kf").glob("*.npz"):
        k = int(f.stem)
        z = np.load(f)
        dr[k] = np.asarray(z["T_dr"], np.float64)[:2, 3].copy()
        dist[k] = float(z["dist_m"])
        if "T_map" in z.files:
            mp[k] = np.asarray(z["T_map"], np.float64)[:2, 3].copy()

    # final_poses.npz 是权威的最终位姿；有就覆盖。
    # ⚠️ **没有它时不能退回逐帧 `T_map`** —— 实测（044153 kf198）逐帧 `T_map` 与最终位姿
    # 差 0.5 追踪米，即 `T_map` 是**写入时刻**的位姿、之后被回环改过而没回写。
    # 拿它算 r = p_map − p_dr 会得到"某段 r 恒定"的假象（A = 0、残差 0）。045615 就是这种情况。
    has_final = final_npz.exists() if final_npz is not None else (rec / "final_poses.npz").exists()
    if has_final:
        z = np.load(final_npz if final_npz is not None else (rec / "final_poses.npz"))
        for i, T in zip(z["ids"], z["T"]):
            mp[int(i)] = np.asarray(T, np.float64)[:2, 3].copy()

    loops: list[dict] = []
    for line in (rec / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("kind") == "kf" and e.get("loops"):
            loops += [dict(L, _b=int(e["k"])) for L in e["loops"]]
    return dr, mp, dist, loops, ws, has_final


def evidence_a(name: str, loops: list[dict], dist: dict[int, float], yaw_tol: float) -> list[float]:
    """证据 A：回环自身的 c/L 与基线的关系。返回可用的比值列表。"""
    rows = []
    for L in loops:
        a, b = int(L["a"]), int(L["b"])
        if a not in dist or b not in dist:
            continue
        if float(L.get("rot_err_deg", 99.0)) > yaw_tol:
            continue
        base = dist[b] - dist[a]
        c = float(L.get("correction_m", 0.0))
        if base <= 1e-6 or c <= 0.0:
            continue
        rows.append((base, c, c / base))

    print(f"  通过转角门（≤{yaw_tol:g}°）且有路程的回环 {len(rows)}/{len(loops)}")
    if not rows:
        return []
    base = np.array([r[0] for r in rows])
    corr = np.array([r[1] for r in rows])
    ratio = corr / base * 100.0
    print(f"  c/L 全样本：中位 {np.median(ratio):.2f}%  p90 {np.percentile(ratio,90):.2f}%  "
          f"最大 {ratio.max():.2f}%")
    print(f"  c   全样本：中位 {np.median(corr)*100:.1f} cm  p90 {np.percentile(corr,90)*100:.1f} cm  "
          f"最大 {corr.max()*100:.1f} cm")
    print("  按两帧间路程分档（c/L 恒定 ⇒ 纯线性漂移；c 恒定 ⇒ 噪声地板，与路程无关）")
    print(f"    {'路程档':>14} {'条数':>5} {'L 中位':>8} {'c 中位':>8} {'c/L 中位':>9} "
          f"{'c p90':>8} {'c 最大':>8}")
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (base >= lo) & (base < hi)
        if int(m.sum()) < 5:
            continue
        tag = f"{lo:4.1f}–{hi:5.1f} m" if hi < 1e8 else f">{lo:4.1f} m"
        print(f"    {tag:>14} {int(m.sum()):5d} {np.median(base[m]):7.1f}m "
              f"{np.median(corr[m])*100:7.1f}cm {np.median(ratio[m]):8.2f}% "
              f"{np.percentile(corr[m],90)*100:7.1f}cm {corr[m].max()*100:7.1f}cm")

    # c = a + b·L：a 是"与路程无关的地板"，b 才是每米的漂移率
    b, a = np.polyfit(base, corr, 1)
    pred = a + b * base
    ss = 1.0 - ((corr - pred) ** 2).sum() / max(((corr - corr.mean()) ** 2).sum(), 1e-12)
    print(f"  拟合 c = a + b·L： 地板 a = {a*100:+.1f} cm，斜率 b = {b*100:.3f} %/m，R² {ss:.3f}")
    if abs(a) > 0.05 and abs(b) < 0.002:
        print("    判读：地板显著、斜率近零 ⇒ 残差主要是**与路程无关的噪声**，不是累积漂移")
    elif b > 0.002:
        print(f"    判读：斜率 {b*100:.2f}%/m 显著 ⇒ 存在**随路程累积**的成分"
              f"（218 m 上折合 {b*218*100:.0f} cm）")
    return [r[2] for r in rows]


def _unique_positions(p: np.ndarray) -> np.ndarray:
    """布尔掩码：**连续重合**的位置只保留第一个（去重口径）。

    为什么必须去重：`kf_max_age_s = 3.0` 意味着**站着不动也每 3 秒补一个关键帧**
    （刷新动态物体用）。关键帧因此**不是按位置均匀取的**：同一个地点可以贡献上千帧，
    而无权最小二乘会把每一帧当成独立样本。实测 `20260929_045615` 的 3014 个关键帧里
    **1451 个（48.1%）落在同一个 `T_dr` 位置上**（唯一位置只有 1563 个）。
    """
    keep = np.ones(len(p), bool)
    if len(p) > 1:
        same = np.all(np.isclose(np.diff(p, axis=0), 0.0, atol=1e-6), axis=1)
        keep[1:] = ~same
    return keep


def _dwell_weights(p: np.ndarray) -> np.ndarray:
    """按 1/驻留数 加权（去重的连续版本，做稳健性对照）。"""
    size: dict[tuple, int] = {}
    for row in p:
        key = tuple(np.round(row, 6))
        size[key] = size.get(key, 0) + 1
    return np.array([1.0 / size[tuple(np.round(row, 6))] for row in p], np.float64)


def evidence_b(name: str, dr: dict[int, np.ndarray], mp: dict[int, np.ndarray],
               dist: dict[int, float], ws: float) -> None:
    """证据 B：位姿场 r = p_map − p_dr 的相似变换分解。

    🔴 **读它之前必须先读这条（2026-10-15，`affine_caliber.py`）**：
    这个「尺度」**不是** DR 尺度误差的读数。受控注入实测（只把 `T_dr` 乘已知倍率、
    回环约束不动）：注入 30% 的已知尺度误差，本量只动 2–6 个百分点 ⇒ **传函增益 0.05–0.11**。
    机制是 `nav_loop.optimize()` 里里程边权 `1/σ_odo²`（0.03 m）比回环边 `1/σ_loop²`（0.1 m）
    硬 ~11 倍且条数远多 ⇒ 地图尺寸跟着 DR 走（`map_rms/dr_rms ≈ 1.00`）。
    它可以在**固定 `σ_odo`** 下用于：① 描述「地图与 DR 之间的修正场」有多大；
    ② 形态判别（反对称主导 vs 对称主导 vs 都小）。
    **不得**用于：推断 DR 链的尺度/朝向误差、比较「哪个场次的 DR 更准」、
    或拿它跨场次排名。要 DR 的独立尺度读数请用 `stereo_geom_chain.py`（不吃 OSC、不吃回环）。
    """
    ids = sorted(set(dr) & set(mp))
    if len(ids) < 8:
        print("  可用关键帧太少，跳过")
        return
    p_all = np.array([dr[k] for k in ids], np.float64) * ws      # 世界米
    q_all = np.array([mp[k] for k in ids], np.float64) * ws
    r_all = q_all - p_all
    path = np.array([dist.get(k, np.nan) for k in ids], np.float64)
    nrm = np.linalg.norm(r_all, axis=1)

    keep = _unique_positions(p_all)
    dup = 1.0 - float(keep.mean())
    print(f"  关键帧 {len(ids)}（T_dr ∩ T_map），world_scale={ws:g}")
    print(f"  |r| = |p_map − p_dr|：中位 {np.median(nrm):.3f} m  "
          f"p90 {np.percentile(nrm,90):.3f}  最大 {nrm.max():.3f}")
    print(f"  位置唯一数 {int(keep.sum())} / {len(ids)}（重合 {dup*100:.1f}%）"
          + ("  ⇒ ⚠️ 站立补帧把同一地点重复计入，无权拟合会放大它 ⇒ 下面同时给 loc 口径"
             if dup > 0.05 else "  ⇒ 关键帧基本按位置散布，all 与 loc 口径差别很小"))

    ok = np.isfinite(path)
    if int(ok.sum()) >= 8 and float(np.ptp(path[ok])) > 1e-6:
        sl, ic = np.polyfit(path[ok], nrm[ok], 1)
        pred = sl * path[ok] + ic
        ss = 1.0 - ((nrm[ok] - pred) ** 2).sum() / max(((nrm[ok] - nrm[ok].mean()) ** 2).sum(), 1e-12)
        print(f"  |r| vs 累计路程：斜率 {sl*100:+.3f} %/m   R² {ss:.3f}"
              f"   （路程 p50 {np.median(path[ok]):.1f} / 最大 {path[ok].max():.1f} m）")

    def _fit(p: np.ndarray, r: np.ndarray, w: np.ndarray | None = None) -> dict:
        """全仿射拟合 r ≈ A·p + t（6 参数），再拆 A。"""
        X = np.zeros((2 * len(p), 6), np.float64)
        y = np.zeros(2 * len(p), np.float64)
        X[0::2, 0], X[0::2, 1], X[0::2, 2] = p[:, 0], p[:, 1], 1.0
        X[1::2, 3], X[1::2, 4], X[1::2, 5] = p[:, 0], p[:, 1], 1.0
        y[0::2], y[1::2] = r[:, 0], r[:, 1]
        if w is None:
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = float(np.sqrt(np.mean((X @ sol - y) ** 2)))
            raw = float(np.sqrt(np.mean(y ** 2)))
        else:
            ww = np.repeat(w, 2)
            sol, *_ = np.linalg.lstsq(X * np.sqrt(ww)[:, None], y * np.sqrt(ww), rcond=None)
            m = ww > 0
            res = float(np.sqrt(np.average((X @ sol - y)[m] ** 2, weights=ww[m])))
            raw = float(np.sqrt(np.average(y ** 2, weights=ww)))
        A = np.array([[sol[0], sol[1]], [sol[3], sol[4]]], np.float64)
        s11 = 0.5 * (A[0, 0] - A[1, 1])
        s12 = 0.5 * (A[0, 1] + A[1, 0])
        return {"sol": sol, "res": res, "raw": raw, "A": A,
                "scale": 0.5 * (A[0, 0] + A[1, 1]), "omega": 0.5 * (A[1, 0] - A[0, 1]),
                "shear": float(np.sqrt(2.0 * (s11 ** 2 + s12 ** 2)))}

    all_fit = _fit(p_all, r_all)
    loc_fit = _fit(p_all[keep], r_all[keep])
    dwe_fit = _fit(p_all, r_all, _dwell_weights(p_all))

    # 只用纯旋转 + 平移（3 参数）的对照（all 口径，保持与历史输出可比）
    Xr = np.zeros((2 * len(ids), 3), np.float64)
    Xr[0::2, 0], Xr[0::2, 1] = -p_all[:, 1], 1.0
    Xr[1::2, 0], Xr[1::2, 2] = p_all[:, 0], 1.0
    yr = np.empty(2 * len(ids), np.float64)
    yr[0::2], yr[1::2] = r_all[:, 0], r_all[:, 1]
    solr, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
    res_rot = float(np.sqrt(np.mean((Xr @ solr - yr) ** 2)))

    print("  ★ r ≈ A·p_dr + t 的分解")
    print(f"    {'口径':<8}{'尺度':>10}{'旋转':>10}{'剪切':>9}{'拟合残差':>11}")
    for tag, f in (("all", all_fit), ("loc", loc_fit), ("dwell", dwe_fit)):
        print(f"    {tag:<8}{f['scale']*100:>+9.2f}%{np.degrees(f['omega']):>+9.3f}°"
              f"{f['shear']*100:>8.2f}%{f['res']*100:>10.1f} cm")
    if dup > 0.05:
        print("    ⇒ all 与 loc 差异>0.1pp 时**以 loc 为准**（all 被站立补帧加权）")
    print(f"    拟合残差 rms（all）： 原样 {np.sqrt(np.mean(yr**2))*100:.1f} cm → "
          f"纯旋转+平移 {res_rot*100:.1f} cm → 全仿射 {all_fit['res']*100:.1f} cm")
    scale, omega = loc_fit["scale"], loc_fit["omega"]
    verdict = []
    if abs(np.degrees(omega)) > 0.5 and abs(scale) < 0.005:
        verdict.append("反对称主导 ⇒ **常量朝向偏移**")
    if abs(scale) >= 0.005:
        verdict.append(f"对称部分不可忽略（尺度 {scale*100:+.2f}%）⇒ 有**尺度**成分")
    if loc_fit["res"] > 0.5 * loc_fit["raw"]:
        verdict.append("拟合后残差仍过半 ⇒ 有**非刚体**成分（随机游走）")
    print(f"    判读（loc 口径）：{'；'.join(verdict) if verdict else '无显著成分，接近噪声'}")
    print("    ⚠️ 上面的『尺度成分』只描述**修正场**的形状。它的传函增益只有 0.05–0.11，"
          "**不能**读成『DR 长了 x%』——见函数 docstring。")

    # 纯旋转要求的角度：把 DR 轨迹整体转 ω 后 r 还剩多少（直接可执行的量）
    ang = np.degrees(omega)
    if abs(ang) > 0.05:
        c, s = np.cos(omega), np.sin(omega)
        R = np.array([[c, -s], [s, c]])
        pr = (R @ p_all.T).T
        res_after = float(np.sqrt(np.mean((pr - q_all) ** 2)))
        print(f"    可执行量：若把用于旋转速度的朝向整体加 {ang:+.3f}°，"
              f"轨迹残差 rms 从 {float(np.sqrt(np.mean((q_all-p_all)**2)))*100:.1f} cm 降到 "
              f"{res_after*100:.1f} cm")


def evidence_c(name: str, mp: dict[int, np.ndarray], loops: list[dict], ws: float) -> None:
    """证据 C：最终地图自己满足不满足它记录下来的回环约束？

    对每条回环，(a,b) 是"同一地点"的两帧，``offset_m`` 是几何链给的两帧距离（世界米）。
    所以最终地图里 ``|p_map[b] − p_map[a]|`` 应当 ≈ ``offset_m``。差多少就是**地图没有满足
    自己那条约束**多少 —— 这正是文档 §九 说的"留出组残差 4.67 m"那一类量，
    与证据 A 的 ``correction_m``（检测时的残差）**不是同一个数**，两者不能互相印证。
    """
    mm = []
    for L in loops:
        a, b = int(L["a"]), int(L["b"])
        if a not in mp or b not in mp:
            continue
        geo = float(L.get("offset_m", 0.0))
        if geo <= 1e-6:
            continue
        mm.append(np.hypot(*(mp[b] - mp[a])) * ws - geo)
    if not mm:
        print("  无可用回环")
        return
    mm = np.abs(np.array(mm))
    print(f"  回环 {len(mm)} 条：地图的两帧距离与几何链之差（绝对值）")
    print(f"    中位 {np.median(mm)*100:.1f} cm   p90 {np.percentile(mm,90)*100:.1f} cm   "
          f"最大 {mm.max()*100:.1f} cm")
    bad = int((mm > 1.0).sum())
    print(f"    差 > 1 m 的 {bad} 条（{100.0*bad/len(mm):.1f}%）")
    if mm.max() > 3.0:
        print("    ⚠️ 存在 >3 m 的未满足约束 ⇒ 回环之间**互相不一致**，不只是 DR 漂移")
    else:
        print("    地图基本满足自己的回环约束（这是**内部自洽**，不等于绝对米制正确）")


def evidence_d(name: str, dr: dict[int, np.ndarray], loops: list[dict],
               dist: dict[int, float], ws: float, min_offset: float = 0.5,
               nbins: int = 4) -> None:
    """证据 D：**把会话按时段切段**，看 OSC/几何 的标度比 s 是恒定还是随会话增长。

    这条直接回答 §3.3 留下的问题：`045615` 的 −10.9% 尺度是
      * 整场恒定的系统性偏差（⇒ 该场的标定 / 录制问题），还是
      * 随时段增长的累积型误差（⇒ 采样保持 / 积分误差）。

    只用回环数据，不依赖 `T_map`；几何位移 ≥ ``min_offset`` 以避开短基线噪声
    （`loop_selfcal.py` 的教训：0.13 m 位移上 5 cm 误差就是 38%）。
    """
    rows: list[tuple[float, float, float]] = []
    for L in loops:
        a, b = int(L["a"]), int(L["b"])
        if a not in dr or b not in dr or b not in dist:
            continue
        geo = float(L.get("offset_m", 0.0))
        if geo < min_offset:
            continue
        base = dist[b] - dist[a]
        if base <= 1e-6:
            continue
        s = float(np.hypot(*(dr[b] - dr[a]))) * ws / geo
        rows.append((float(dist[b]), s, geo))

    if len(rows) < 12:
        print(f"  几何位移 ≥{min_offset} m 的回环只有 {len(rows)} 条，样本不足")
        return
    rows.sort()
    prog = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows])
    geo = np.array([r[2] for r in rows])
    tot = max(prog.max(), 1e-6)
    print(f"  几何位移 ≥{min_offset} m 的回环 {len(rows)} 条（去掉短基线噪声）；"
          f"s = |OSC 位移| / 几何位移")
    print(f"    s 全样本 中位 {np.median(s):.4f}  IQR "
          f"[{np.percentile(s,25):.4f}, {np.percentile(s,75):.4f}]"
          f"  几何基线 中位 {np.median(geo):.2f} m")
    # ★ 显著性：中位数离 1 有多远？长基线样本极少（十几到几十条），点估计**不能当结论**。
    rng = np.random.default_rng(20261005)
    bs = np.array([np.median(rng.choice(s, s.size, replace=True)) for _ in range(2000)])
    lo, hi = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
    print(f"    bootstrap 95% CI [{lo:.4f}, {hi:.4f}] —— "
          + ("**含 1.0 ⇒ 与「两链一致」不可区分**（点估计再接近 1 也不能说『一致到 x%』）"
             if lo <= 1.0 <= hi else
             "**不含 1.0 ⇒ 显著偏离 1**"))
    print("    ⚠️ 必须同时看每档的**几何基线**：s 在短基线上虚高，若各档基线不同，"
          "分档差异就是混淆而不是趋势")
    edges = np.linspace(0.0, tot, nbins + 1)
    med_bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prog >= lo) & (prog < hi)
        if int(m.sum()) < 4:
            med_bins.append(None)
            print(f"    会话 {lo:6.0f}–{hi:6.0f} m: {int(m.sum()):3d} 条  (样本不足，跳过)")
            continue
        med_bins.append(float(np.median(s[m])))
        print(f"    会话 {lo:6.0f}–{hi:6.0f} m（{100*lo/tot:3.0f}–{100*hi/tot:3.0f}%）: "
              f"{int(m.sum()):3d} 条   s 中位 {np.median(s[m]):.4f}   "
              f"几何基线 中位 {np.median(geo[m]):.2f} m")
    good = [v for v in med_bins if v is not None]
    if len(good) >= 3:
        spread = max(good) - min(good)
        print(f"    分段极差 {spread*100:.2f} 个百分点 —— "
              + ("接近恒定 ⇒ 整场系统性偏差（标定 / 录制问题）"
                 if spread < 0.10 else
                 "随时段变化 ⇒ 需先排除上面的基线混淆，再谈累积成分"))


def evidence_e(name: str, dr: dict[int, np.ndarray], mp: dict[int, np.ndarray],
               ws: float, nbins: int = 4) -> None:
    """证据 E：**按会话进度切段**，逐段做仿射分解——比证据 D 功率高两个数量级。

    证据 D 用回环（长基线下只剩 13/13/4 条），而且 ``s`` 不是纯尺度量：它是"DR 直线位移 /
    几何位移"，长路程上累积的**朝向**误差同样会污染它。这里改用全部关键帧：把 ``r = p_map − p_dr``
    按顺序切段，每段给**自己的平移**（段心对齐，避免病态），比较各段 ``A`` 的
    尺度 / 旋转 / 剪切分量。

    判据：各段分量**基本一致** ⇒ 整场系统性偏差（标定 / 录制）；
    随进度**单调变化** ⇒ 累积型误差。
    """
    ids = sorted(set(dr) & set(mp))
    if len(ids) < nbins * 40:
        print(f"  关键帧只有 {len(ids)}，不足以切 {nbins} 段")
        return
    p_all = np.array([dr[k] for k in ids], np.float64) * ws
    q_all = np.array([mp[k] for k in ids], np.float64) * ws
    # ★ 与证据 B 同一处修正：`kf_max_age_s` 让站着不动也每 3 s 补帧，各段的"帧数"因此
    #   不等于"时长/行程"。045615 的末段就是纯冻结簇（1250 帧同一个 T_dr），旧口径给出
    #   「尺度 −0.00%、残差 0.0 cm」的退化段（§9.3 记过这个现象，当时归因为"没被回环碰过"，
    #   其实主因是重复帧）。切段前先按位置去重。
    keep = _unique_positions(p_all)
    p, q, r = p_all[keep], q_all[keep], q_all[keep] - p_all[keep]
    n = len(p)
    print(f"  关键帧 {len(ids)} → 按位置去重 {n}（去掉 {len(ids)-n} 个重复位置），"
          f"再按顺序切 {nbins} 段（每段自己的平移，只比 A 的分量）")
    rows = []
    for i in range(nbins):
        lo, hi = i * n // nbins, (i + 1) * n // nbins
        ps, rs = p[lo:hi], r[lo:hi]
        d = ps - ps.mean(axis=0)
        X = np.zeros((2 * len(ps), 6), np.float64)
        y = np.zeros(2 * len(ps), np.float64)
        X[0::2, 0], X[0::2, 1], X[0::2, 2] = d[:, 0], d[:, 1], 1.0
        X[1::2, 3], X[1::2, 4], X[1::2, 5] = d[:, 0], d[:, 1], 1.0
        y[0::2], y[1::2] = rs[:, 0], rs[:, 1]
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        A = np.array([[sol[0], sol[1]], [sol[3], sol[4]]], np.float64)
        scale = 0.5 * (A[0, 0] + A[1, 1])
        omega = 0.5 * (A[1, 0] - A[0, 1])
        s11 = 0.5 * (A[0, 0] - A[1, 1])
        s12 = 0.5 * (A[0, 1] + A[1, 0])
        shear = float(np.sqrt(2.0 * (s11 ** 2 + s12 ** 2)))
        res = float(np.sqrt(np.mean((X @ sol - y) ** 2)))
        med_r = float(np.median(np.linalg.norm(rs, axis=1)))
        rows.append((scale, np.degrees(omega), shear))
        print(f"    段{i+1}（帧 {lo:5d}–{hi:5d}）：尺度 {scale*100:+7.2f}%   "
              f"旋转 {np.degrees(omega):+7.3f}°   剪切 {shear*100:5.2f}%   "
              f"|r| 中位 {med_r*100:5.1f} cm   拟合残差 {res*100:5.1f} cm")
    sc = [x[0] for x in rows]
    ro = [x[1] for x in rows]
    print(f"    分段极差： 尺度 {100*(max(sc)-min(sc)):.2f} 个百分点，"
          f"旋转 {max(ro)-min(ro):.3f}°")
    if (max(sc) - min(sc)) < 0.02 and (max(ro) - min(ro)) < 1.0:
        print("    判读：各段接近一致 ⇒ **整场系统性偏差**（标定 / 录制问题），不是累积")
    else:
        print("    判读：各段不一致 ⇒ 有**随会话变化**的成分（看上面是单调还是抖动）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="*", default=["20261001_044153", "20260929_045615"])
    ap.add_argument("--yaw-tol", type=float, default=6.0)
    ap.add_argument("--min-offset", type=float, default=1.5,
                    help="证据 D 用的几何基线下限（世界米）；1.5 是 loop_selfcal 里最干净的档")
    ap.add_argument("--final-poses-dir", type=Path, default=None,
                    help="外部最终位姿目录，找 <dir>/<录制名>.npz（`loop_replay.py` 的产物）；"
                         "用于没有 final_poses.npz 的老录制。⚠️ 只有该回放通过自检才可信。")
    args = ap.parse_args()

    for name in args.recs:
        rec = ROOT / "navmesh_recordings" / name
        if not rec.exists():
            print(f"[skip] {name}", file=sys.stderr)
            continue
        alt = (args.final_poses_dir / f"{name}.npz") if args.final_poses_dir else None
        dr, mp, dist, loops, ws, has_final = load(rec, alt)
        src = "final_poses.npz（录制自带）" if has_final and alt is None else (
            f"{alt}（`loop_replay.py` 产物，需自检）" if has_final else "❌缺")
        print(f"\n=== {name} ===  回环 {len(loops)} 条，关键帧 T_dr {len(dr)} / T_map {len(mp)}"
              f"   最终位姿：{src}")
        print("  ── 证据 A：回环自身 ──")
        evidence_a(name, loops, dist, args.yaw_tol)
        if not has_final:
            print("  ── 证据 B/C/E：❌ **跳过** ──")
            print("     该录制没有 `final_poses.npz`，逐帧 `T_map` 是**写入时刻**的位姿"
                  "（之后被回环改过、没回写），拿它算 r 会得到「某段 r 恒定」的假象。")
            print("     ⇒ 本录制**无法**用位姿场判形态；只有证据 A / D 可用（都只用 T_dr 与回环记录）。")
        else:
            print("  ── 证据 B：位姿场（回环修正场） ──")
            evidence_b(name, dr, mp, dist, ws)
            print("  ── 证据 C：地图是否满足自己的回环约束 ──")
            evidence_c(name, mp, loops, ws)
        print("  ── 证据 D：按时段切段，标度比 s 是恒定还是累积 ──")
        evidence_d(name, dr, loops, dist, ws, min_offset=args.min_offset)
        if has_final:
            print("  ── 证据 E：按会话进度切段，逐段仿射分解（满功率） ──")
            evidence_e(name, dr, mp, ws)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
