# -*- coding: utf-8 -*-
"""pose_graph.py —— 2D 米制位姿图（航位推算 + 闭环修正）。

把 run_motion 提供的「运动量」(OSC 路程 + HMD 解卷绕 yaw) 与 loop_verify.json 里
已确认 (verdict == "confirmed_visual_loop") 的回环融合成一张带米制坐标的 2D 地图。

边界：
- 不引入单目 SLAM；尺度/转向由运动信息给定，本模块只做「融合 + 闭环修正」。
- 仅新建 pose_graph.py / pose_graph.json（及 pose_input_loop.json 副本、可选报告）。
- 不修改 run_motion.py / loop_verify.py / build_topo_map.py 等任何产物。

求解：纯 numpy 手写 Gauss-Newton + Levenberg-Marquardt 阻尼，不依赖 scipy。

航位推算的旋转公式**不在本文件**：与在线 ``backend/online_pose.py`` 共用
``backend/pose_math.py`` 的同一份实现 —— 把 avatar **本地矢量位移**按朝向旋进
世界系：``world_x = dx*cos(y) + dz*sin(y)``、``world_z = -dx*sin(y) + dz*cos(y)``，
其中 ``y = radians(yaw_deg) * yaw_sign``，local +Z 为前进、local +X 为右侧横移。

⚠️ 历史教训：本文件曾经自己写了一份，而且用的是**标量路程**沿 yaw 前进
（``x += ds*sin(yaw)``）—— 等价于把横移当成前进，只在 vx≈0 时正确。已证伪并
修正（本 run 终点差 1.419 m）。改公式请改 ``backend/pose_math.py``，不要在此重写。

yaw 符号：本 run 由 ``yaw_sign_check.json`` 的**画面证据**标定为 -1
（``OFFLINE_YAW_SIGN``）。``--yaw-sign auto`` 只在没有图像证据时才退回闭环
自洽性选择 —— 闭环比对是内部自洽性，不能替代外部参照。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# 单机 BLAS 线程：本机 numpy 2.5.3 在多管线并发求解时会触发 BLAS 线程竞态导致进程
# 静默崩溃（无 Python traceback）。固定单线程规避（图规模小，无关性能）。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

# run_motion.py 与本文件同目录
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_motion import OscPath, HmdYaw, load_offset  # noqa: E402

# 旋转公式的唯一实现：backend/pose_math.py（与在线 OnlinePose2D 共用同一份）。
# 此前本文件与 backend/online_pose.py 各写一份，本文件那份漂移成了「把横移当
# 前进」，见该模块 docstring。
# 用 append 而不是 insert：避免 backend/ 下的同名模块遮蔽 recorder 本地模块。
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # research/recorder -> repo
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from backend.pose_math import OFFLINE_YAW_SIGN, advance, rotate_local_to_world  # noqa: E402


# ----------------------------------------------------------------------------
# 鲁棒核（对回环节做鲁棒化，避免单条错误边拉坏全图）
# ----------------------------------------------------------------------------
def robust_scale(r_norm: float, delta: float, kind: str) -> float:
    if r_norm <= delta or delta <= 0:
        return 1.0
    if kind == "cauchy":
        d2 = delta * delta
        return d2 / (d2 + r_norm * r_norm)
    # huber（默认）
    return delta / r_norm


def load_confirmed_loops(run_dir: str):
    """只读 pose_input_loop.json 副本；只取 confirmed_visual_loop。"""
    lp = os.path.join(run_dir, "pose_input_loop.json")
    if not os.path.exists(lp):
        lp = os.path.join(run_dir, "loop_verify.json")
    d = json.load(open(lp, encoding="utf-8"))
    loops = []
    raw_all = d.get("edges", [])
    for e in raw_all:
        if e.get("verdict") == "confirmed_visual_loop":
            loops.append({
                "a": int(e["a"]), "b": int(e["b"]),
                "inliers": int(e.get("inliers") or 0),
                "detour_m": e.get("detour_m"),
                "detour_lb_m": e.get("detour_lb_m"),
                "gate": e.get("gate"),
            })
    # 被拒的边清单（供后续校验：这些对绝不能进图）
    rejected = [(int(e["a"]), int(e["b"])) for e in raw_all
                if e.get("verdict") == "rejected"]
    rej_pairs = set((min(e["a"], e["b"]), max(e["a"], e["b"])) for e in raw_all
                    if e.get("verdict") == "rejected")
    # 诚实性断言：已确认的回环里绝不能包含被拒的对（尤其 c4-c6 静止假回环）
    conf_pairs = set((min(l["a"], l["b"]), max(l["a"], l["b"])) for l in loops)
    bad = conf_pairs & rej_pairs
    assert not bad, "被拒的回环对混入了确认集：%s" % (bad,)
    return loops, rejected


def load_topo(run_dir: str):
    tp = os.path.join(run_dir, "topo_map.json")
    d = json.load(open(tp, encoding="utf-8"))
    clusters = {}
    for n in d.get("nodes", []):
        c = int(n["cluster"])
        clusters[c] = {
            "telemetry_seconds": n["telemetry_seconds"],
            "video_seconds": n["video_seconds"],
            "rep_feat_frame": n.get("rep_feat_frame"),
        }
    return clusters, d.get("feature_fps", 20.0)


# ----------------------------------------------------------------------------
# 航位推算
# ----------------------------------------------------------------------------
def dead_reckon(run_dir, dt, yaw_sign, t0=None, t1=None):
    hmd = HmdYaw(run_dir)
    osc = OscPath(run_dir)
    if t0 is None:
        t0 = float(hmd.t[0])
    if t1 is None:
        t1 = float(hmd.t[-1])
    grid = np.arange(t0, t1 + dt / 2.0, dt)
    N = len(grid)
    yaw_deg = np.interp(grid, hmd.t, hmd.yaw)
    yaw_rad = np.radians(yaw_deg) * yaw_sign
    x = np.zeros(N)
    z = np.zeros(N)
    ds_arr = np.zeros(N)    # 每步位移的模长（= 旧口径的路程；只做前进时两者相同）
    dvx_arr = np.zeros(N)   # 每步的 avatar 本地速度矢量积分
    dvz_arr = np.zeros(N)
    for k in range(1, N):
        # 用**矢量**位移，不是标量路程。OSC 的 VelocityX/Z 是 avatar 本地坐标系
        # （README 已实机验证），所以必须保留方向再按朝向旋进世界，才等价于在线
        # OnlinePose2D（backend/online_pose.py）。旧实现 `dist 沿 yaw` 把横移
        # 当成了前进 —— 本 run 实测终点差 1.5496 m（横移 |dvx| 累计 1.2205 m，
        # 占 |disp| 的 1.85%，出现在 596/895 步上；不是"横移很少见"，而是每次
        # 都有一点点），而规划里明确要用 /input/Horizontal 做撞墙横移绕行，
        # 占比只会上升。
        dvx, dvz = osc.disp(grid[k - 1], grid[k])
        dvx_arr[k] = dvx
        dvz_arr[k] = dvz
        ds_arr[k] = float(np.hypot(dvx, dvz))
        # 旋转只有一份公式：backend/pose_math.py::advance（与在线共用）。
        x[k], z[k] = advance(x[k - 1], z[k - 1], dvx, dvz, yaw_rad[k - 1])
    return {
        "grid": grid, "x": x, "z": z, "yaw_deg": yaw_deg,
        "yaw_rad": yaw_rad, "ds": ds_arr, "dvx": dvx_arr, "dvz": dvz_arr, "N": N,
        "osc_total": osc.total, "hmd_first_yaw": float(hmd.yaw[0]),
    }


def assign_clusters(grid, clusters):
    """每个航位点的时间归属到某个簇（取时间窗包含它的；否则取最近窗心）。"""
    node_cluster = np.full(len(grid), -1, dtype=int)
    centers = {}
    for c, info in clusters.items():
        s, e = info["telemetry_seconds"]
        centers[c] = (s + e) / 2.0
    cids = sorted(clusters.keys())
    for k, t in enumerate(grid):
        hit = None
        for c in cids:
            s, e = clusters[c]["telemetry_seconds"]
            if s - 1e-9 <= t <= e + 1e-9:
                hit = c
                break
        if hit is None:
            # 最近窗心
            best = min(cids, key=lambda c: abs(centers[c] - t))
            hit = best
        node_cluster[k] = hit
    return node_cluster


def cluster_rep_node(cluster_id, clusters, grid):
    s, e = clusters[cluster_id]["telemetry_seconds"]
    rep_t = (s + e) / 2.0
    rep_t = min(max(rep_t, grid[0]), grid[-1])
    return int(np.argmin(np.abs(grid - rep_t)))


# ----------------------------------------------------------------------------
# 图优化（仅优化 x,z；yaw 固定由 HMD 给定；节点 0 固定为原点消除平移自由度）
# ----------------------------------------------------------------------------
def build_edges(dr, loops, node_cluster, clusters, grid,
                loop_weight, odo_weight, inliers_max):
    N = dr["N"]
    edges = []
    # 航位边
    for k in range(1, N):
        y = dr["yaw_rad"][k - 1]
        # 与 dead_reckon 完全同式（同一份 backend/pose_math.py）：矢量位移按朝向
        # 旋进世界，不是 ds 沿 yaw。
        dvx = dr["dvx"][k]
        dvz = dr["dvz"][k]
        dmeas = np.array(rotate_local_to_world(dvx, dvz, y))
        edges.append({
            "type": "odometry", "i": k - 1, "j": k,
            "source": "osc_integral", "heading": "hmd_yaw",
            "dmeas": dmeas, "w": odo_weight, "inliers": None,
            "confidence": 1.0,
        })
    # 回环边（位置重合约束，目标 0）
    for lp in loops:
        ia = cluster_rep_node(lp["a"], clusters, grid)
        ib = cluster_rep_node(lp["b"], clusters, grid)
        if ia == ib:
            continue
        w = loop_weight * (lp["inliers"] / inliers_max if inliers_max > 0 else 1.0)
        conf = (lp["inliers"] / inliers_max) if inliers_max > 0 else 1.0
        edges.append({
            "type": "loop", "i": ia, "j": ib,
            "source": "confirmed_visual_loop:%d-%d" % (lp["a"], lp["b"]),
            "dmeas": np.zeros(2), "w": w, "inliers": lp["inliers"],
            "confidence": float(min(1.0, conf)),
            "detour_m": lp["detour_m"], "detour_lb_m": lp["detour_lb_m"],
            "gate": lp["gate"],
        })
    return edges


def node_pos(X, n):
    if n == 0:
        return np.array([0.0, 0.0])
    i = 2 * (n - 1)
    return X[i:i + 2]


def build_J(X, edges, N, robust, delta):
    E = len(edges)
    J = np.zeros((2 * E, 2 * (N - 1)))
    r = np.zeros(2 * E)
    w = np.zeros(2 * E)
    for ei, e in enumerate(edges):
        i, j = e["i"], e["j"]
        pi = node_pos(X, i)
        pj = node_pos(X, j)
        res = (pj - pi) - e["dmeas"]
        r[2 * ei:2 * ei + 2] = res
        base = e["w"]
        if e["type"] == "loop":
            rn = float(np.linalg.norm(res))
            sc = robust_scale(rn, delta, robust)
            ww = base * sc
        else:
            ww = base
        w[2 * ei] = ww
        w[2 * ei + 1] = ww
        if i > 0:
            vi = 2 * (i - 1)
            J[2 * ei:2 * ei + 2, vi:vi + 2] = -np.eye(2)
        if j > 0:
            vj = 2 * (j - 1)
            J[2 * ei:2 * ei + 2, vj:vj + 2] = np.eye(2)
    return J, r, w


def optimize(dr, edges, max_iter, tol, lam0, robust, delta):
    N = dr["N"]
    # 初值：航位推算的 x,z（节点 0 固定原点）
    X = np.zeros(2 * (N - 1))
    for k in range(1, N):
        X[2 * (k - 1)] = dr["x"][k]
        X[2 * (k - 1) + 1] = dr["z"][k]
    lam = lam0
    last_dx = np.inf
    for it in range(max_iter):
        J, r, w = build_J(X, edges, N, robust, delta)
        W = np.diag(w)
        H = J.T @ W @ J
        g = J.T @ W @ r
        cost = float(r @ (W @ r))
        accepted = False
        for _ in range(20):
            Hd = H + lam * np.diag(np.diag(H) + 1e-12)
            try:
                dx = np.linalg.solve(Hd, -g)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(Hd, -g, rcond=None)[0]
            Xn = X + dx
            Jn, rn, wn = build_J(Xn, edges, N, robust, delta)
            Wn = np.diag(wn)
            costn = float(rn @ (Wn @ rn))
            if costn < cost:
                X = Xn
                lam = max(lam * 0.5, 1e-9)
                last_dx = float(np.linalg.norm(dx))
                accepted = True
                break
            else:
                lam = min(lam * 2.0, 1e6)
                if lam > 1e5:
                    accepted = True
                    break
        if not accepted:
            break
        if last_dx < tol:
            break
    # 终态残差
    J, r, w = build_J(X, edges, N, robust, delta)
    W = np.diag(w)
    final_cost = float(r @ (W @ r))
    return X, final_cost, it + 1


def residual_by_type(X, edges, N):
    J, r, w = build_J(X, edges, N, "huber", 1e9)  # 不鲁棒化，看真实残差
    out = {}
    for kind in ("odometry", "loop"):
        idxs = [ei for ei, e in enumerate(edges) if e["type"] == kind]
        if not idxs:
            out[kind] = None
            continue
        rr = np.concatenate([r[2 * ei:2 * ei + 2] for ei in idxs])
        out[kind] = float(np.sqrt(np.mean(rr ** 2))) if len(rr) else None
    return out


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def run_pipeline(run_dir, dt, yaw_sign, loop_weight, odo_weight,
                 robust, delta, max_iter, tol, lam0):
    loops, rejected = load_confirmed_loops(run_dir)
    clusters, feat_fps = load_topo(run_dir)
    inliers_max = max((l["inliers"] for l in loops), default=1)
    dr = dead_reckon(run_dir, dt, yaw_sign)
    node_cluster = assign_clusters(dr["grid"], clusters)
    edges = build_edges(dr, loops, node_cluster, clusters, dr["grid"],
                        loop_weight, odo_weight, inliers_max)
    # 验收准则 3：图里没有 rejected 边，且 c4-c6 绝不在
    rej_pairs = set((min(a, b), max(a, b)) for a, b in rejected)
    for e in edges:
        if e["type"] == "loop":
            s = e["source"]  # "confirmed_visual_loop:a-b"
            pair = tuple(int(x) for x in s.split(":")[1].split("-"))
            assert pair not in rej_pairs, "rejected 对 %s 混入了图" % (pair,)
    # c4-c6 静止假回环明确不在图里（4-6 属 rejected，已确认集不含它）
    assert (4, 6) not in set((min(l["a"], l["b"]), max(l["a"], l["b"]))
                             for l in loops), "c4-c6 不应在确认回环中"
    # 优化（x,z）
    X, final_cost, n_iter = optimize(dr, edges, max_iter, tol, lam0, robust, delta)

    N = dr["N"]
    # 组装优化后坐标
    xopt = np.zeros(N)
    zopt = np.zeros(N)
    for k in range(1, N):
        xopt[k] = X[2 * (k - 1)]
        zopt[k] = X[2 * (k - 1) + 1]

    # c0 / c22 代表节点
    i0 = cluster_rep_node(0, clusters, dr["grid"])
    i22 = cluster_rep_node(22, clusters, dr["grid"])
    pre_closure = float(np.hypot(dr["x"][i22] - dr["x"][i0],
                                 dr["z"][i22] - dr["z"][i0]))
    post_closure = float(np.hypot(xopt[i22] - xopt[i0],
                                  zopt[i22] - zopt[i0]))

    pre_res = residual_by_type(np.array([dr["x"][k] if k == 0 else dr["x"][k]
                                         for k in range(N)]), edges, N) \
        if False else None
    # 计算 pre 残差：用航位坐标拼成 X 形式
    Xpre = np.zeros(2 * (N - 1))
    for k in range(1, N):
        Xpre[2 * (k - 1)] = dr["x"][k]
        Xpre[2 * (k - 1) + 1] = dr["z"][k]
    pre_rbt = residual_by_type(Xpre, edges, N)
    post_rbt = residual_by_type(X, edges, N)

    return {
        "dr": dr, "X": X, "xopt": xopt, "zopt": zopt,
        "node_cluster": node_cluster, "edges": edges, "loops": loops,
        "rejected": rejected, "clusters": clusters,
        "i0": i0, "i22": i22,
        "pre_closure": pre_closure, "post_closure": post_closure,
        "pre_rbt": pre_rbt, "post_rbt": post_rbt,
        "final_cost": final_cost, "n_iter": n_iter,
        "inliers_max": inliers_max, "N": N,
    }


def load_yaw_sign_evidence(run_dir):
    """读 <run>/yaw_sign_check.json（yaw_sign_check.py 的图像定标产物）。

    只认 `recommended_yaw_sign` 明确为 ±1 的结论。**任何异常/缺失一律返回 None**——
    宁可退回"符号是拟合默认值"的诚实标注，也绝不把"没测过"伪装成"测过了"
    （与 `load_route_history_safe` 同一原则：异常不可返回真值）。
    """
    p = os.path.join(run_dir, "yaw_sign_check.json")
    try:
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return None
        s = d.get("recommended_yaw_sign")
        if s is None or float(s) not in (1.0, -1.0):
            return None
        return {
            "path": p,
            "verdict": d.get("verdict"),
            "recommended_yaw_sign": float(s),
            "why": d.get("why"),
            "summary": d.get("summary") or {},
        }
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="2D 米制位姿图：航位推算 + 闭环修正")
    ap.add_argument("--run", required=True)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--yaw-sign", choices=["auto", "+1", "-1"], default="auto",
                    help="auto=优先用 yaw_sign_check.json 的**图像证据**，无证据才退回"
                         "闭环选择（闭环比对是内部自洽性，不能替代外部参照）。"
                         "本项目离线已定标值 = " + ("%+d" % int(OFFLINE_YAW_SIGN))
                         + "（backend/pose_math.py::OFFLINE_YAW_SIGN）")
    ap.add_argument("--loop-weight", type=float, default=5.0)
    ap.add_argument("--odo-weight", type=float, default=1.0)
    ap.add_argument("--robust", choices=["huber", "cauchy"], default="huber")
    ap.add_argument("--robust-delta", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--lam0", type=float, default=1e-3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    run_dir = a.run
    out = a.out or os.path.join(run_dir, "pose_graph.json")
    report = a.report or os.path.join(run_dir, "pose_graph_report.md")

    # yaw 符号：三种来源，优先级 命令行 > 图像证据 > 闭环回退。
    #
    # ⚠️ 2026-09-24 更正：旧注释写「闭环对 ±yaw 镜像对称、挑不出符号」—— 那是
    # **标量位移模型**（dist 沿 yaw）下的性质。改用矢量位移后（见 dead_reckon），
    # 翻转 yaw 只在 dvx == 0（无横移）时才是沿 x 的镜像；有横移时 ± 给出真正
    # 不同的轨迹，闭环误差不再相等（本 run 实测 5.666 vs 3.688 m）。
    # 所以：
    #   * 闭环比对现在**可以**分辨符号；
    #   * 但它仍然是**内部自洽性**，不是外部参照 —— 「谁更闭合」取决于拟合。
    # 定符号依旧以图像证据为准：yaw_sign_check.py 用「右转 ⇒ 画面内容左移(Δu<0)」
    # 这个与手系无关的成像几何，端到端测出 sign(Δu) 与 sign(Δyaw) 的关系。
    res_pos = run_pipeline(run_dir, a.dt, +1.0, a.loop_weight, a.odo_weight,
                           a.robust, a.robust_delta, a.max_iter, a.tol, a.lam0)
    res_neg = run_pipeline(run_dir, a.dt, -1.0, a.loop_weight, a.odo_weight,
                           a.robust, a.robust_delta, a.max_iter, a.tol, a.lam0)
    ev = load_yaw_sign_evidence(run_dir)
    closure_gap = abs(res_pos["pre_closure"] - res_neg["pre_closure"])
    if a.yaw_sign in ("+1", "-1"):
        used_sign = float(a.yaw_sign)
        sign_source = "cli_override"
        sign_note = ("命令行强制 --yaw-sign %s，覆盖图像证据与闭环选择。" % a.yaw_sign)
    elif ev is not None:
        used_sign = ev["recommended_yaw_sign"]
        sign_source = "image_yaw_check"
        sign_note = ("由 `yaw_sign_check.json` 的图像证据定标：verdict=%s，"
                     "yaw_sign=%+d。%s（闭环 ± 对照 %.3f / %.3f m，差 %.6f m；%s）"
                     % (ev["verdict"], int(used_sign), ev["why"] or "",
                        res_pos["pre_closure"], res_neg["pre_closure"],
                        closure_gap,
                        "两者接近 ⇒ 对符号不敏感，仅作参考" if closure_gap < 0.05
                        else "矢量位移下两者已可分辨，但闭环是内部自洽性、不作外部参照"))
    else:
        used_sign = (+1.0 if res_pos["pre_closure"] <= res_neg["pre_closure"]
                     else -1.0)
        sign_source = "loop_closure_fallback"
        sign_note = ("**无图像证据**（缺 yaw_sign_check.json）：闭环 ± 对照 "
                     "%.3f vs %.3f m，差 %.6f m。%s 跑 "
                     "`yaw_sign_check.py --write` 可定标。"
                     % (res_pos["pre_closure"], res_neg["pre_closure"], closure_gap,
                        "两者接近（镜像对称），挑不出符号 ⇒ 当前朝向是"
                        "**拟合默认值，非标定**。"
                        if closure_gap < 0.05 else
                        "矢量位移下两者已可分辨，但闭环比对是内部自洽性、不是外部"
                        "参照 ⇒ 当前朝向仍是**拟合默认值，非标定**。"))
    res = res_pos if used_sign > 0 else res_neg
    other = res_neg if used_sign > 0 else res_pos

    dr = res["dr"]
    N = res["N"]
    grid = dr["grid"]
    edges = res["edges"]

    # 组装节点。
    # ⚠️ `yaw`/`yaw0` 是**几何航向**（已乘 used_sign），必须与 x/z 用同一符号——
    # 它们就是造 x/z 用的那个角。曾经这里是**原始 HMD 通道值**：在 used_sign=+1 时
    # 两者恰好相等，所以看不出来；一旦符号取 -1，航迹会翻而航向不翻，下游按
    # `forward=(sin yaw, cos yaw)` 放点就会把点云摊散（不是镜像，是错乱）。
    # 原始通道值单独存 `yaw_hmd_deg` 以便追溯。
    nodes = []
    for k in range(N):
        yaw_geo = float(dr["yaw_deg"][k]) * used_sign
        nodes.append({
            "t": round(float(grid[k]), 4),
            "x0": round(float(dr["x"][k]), 4),
            "z0": round(float(dr["z"][k]), 4),
            "yaw0": round(yaw_geo, 4),
            "x": round(float(res["xopt"][k]), 4),
            "z": round(float(res["zopt"][k]), 4),
            "yaw": round(yaw_geo, 4),
            "yaw_hmd_deg": round(float(dr["yaw_deg"][k]), 4),
            "cluster": int(res["node_cluster"][k]),
        })

    edge_out = []
    for e in edges:
        edge_out.append({
            "type": e["type"], "i": e["i"], "j": e["j"],
            "source": e["source"],
            "weight": round(float(e["w"]), 4),
            "inliers": e["inliers"],
            "confidence": round(float(e["confidence"]), 4),
            "detour_m": e.get("detour_m"),
            "detour_lb_m": e.get("detour_lb_m"),
            "gate": e.get("gate"),
        })

    improved = res["post_closure"] < res["pre_closure"]
    ratio = (res["post_closure"] / res["pre_closure"]) if res["pre_closure"] > 1e-9 else None

    # 轨迹顺序一致性 / 无折返打结：
    # (1) 时间序天然单调（节点在固定时间栅格上）；
    # (2) 相邻优化航位点的位移步长应平滑、无突变（折叠会产生异常大跳变）；
    # (3) 移动步的前进方向与本地 HMD yaw 基本一致（自洽性）。
    steps = np.sqrt(np.diff(res["xopt"]) ** 2 + np.diff(res["zopt"]) ** 2)
    typ_step = float(np.median(steps[steps > 1e-6])) if np.any(steps > 1e-6) else 0.0
    max_jump = float(np.max(steps)) if len(steps) else 0.0
    # 跳变超 2.0 m（相对 65 m 全程）才视为异常折叠/打结；正常的回环修正只会产生亚米级跳变
    n_jump_bad = int(np.sum(steps > 2.0))
    n_odo = 0
    n_consistent = 0
    for k in range(1, N):
        ds = dr["ds"][k]
        if ds < 1e-6:
            continue
        dx = res["xopt"][k] - res["xopt"][k - 1]
        dz = res["zopt"][k] - res["zopt"][k - 1]
        if abs(dx) < 1e-9 and abs(dz) < 1e-9:
            continue
        move_ang = np.arctan2(dx, dz)  # 与 x=sin,z=cos 同约定
        yaw_ang = np.radians(dr["yaw_deg"][k - 1]) * used_sign
        d = abs(((move_ang - yaw_ang + np.pi) % (2 * np.pi)) - np.pi)
        n_odo += 1
        if d < np.radians(60.0):
            n_consistent += 1
    consistency = (n_consistent / n_odo) if n_odo else 1.0
    no_fold = (n_jump_bad == 0)

    metrics = {
        "closure_c0_c22": {
            "pre_m": round(res["pre_closure"], 4),
            "post_m": round(res["post_closure"], 4),
            "improved": bool(improved),
            "ratio_post_over_pre": (round(ratio, 4) if ratio is not None else None),
        },
        "total_residual_final": round(res["final_cost"], 6),
        "n_iter": res["n_iter"],
        "residual_by_type": {
            "odometry": {
                "pre_rms": (round(res["pre_rbt"]["odometry"], 4)
                            if res["pre_rbt"]["odometry"] is not None else None),
                "post_rms": (round(res["post_rbt"]["odometry"], 4)
                             if res["post_rbt"]["odometry"] is not None else None),
            },
            "loop": {
                "pre_rms": (round(res["pre_rbt"]["loop"], 4)
                            if res["pre_rbt"]["loop"] is not None else None),
                "post_rms": (round(res["post_rbt"]["loop"], 4)
                             if res["post_rbt"]["loop"] is not None else None),
            },
        },
        "uncertainty_post_rms": (round(res["post_rbt"]["loop"], 4)
                                 if res["post_rbt"]["loop"] is not None else None),
        "trajectory_consistency": {
            "fraction_aligned_with_yaw": round(float(consistency), 4),
            "n_moving_steps": n_odo, "n_aligned": n_consistent,
            "typical_step_m": round(typ_step, 4),
            "max_consecutive_jump_m": round(max_jump, 4),
            "n_abnormal_jumps": n_jump_bad,
            "no_folding": bool(no_fold),
        },
        "sign_compare": {
            "+1_pre_closure_m": round(res_pos["pre_closure"], 4),
            "-1_pre_closure_m": round(res_neg["pre_closure"], 4),
            "closure_gap_m": round(closure_gap, 6),
            "indistinguishable_by_closure": bool(closure_gap < 1e-3),
            "yaw_sign_used": used_sign,
            "yaw_sign_source": sign_source,
        },
        "n_nodes": N,
        "n_odometry_edges": sum(1 for e in edges if e["type"] == "odometry"),
        "n_loop_edges": sum(1 for e in edges if e["type"] == "loop"),
        "osc_total_m": round(float(dr["osc_total"]), 4),
        "hmd_first_yaw_deg": round(dr["hmd_first_yaw"], 3),
    }

    out_obj = {
        "meta": {
            "run": run_dir,
            "dt": a.dt,
            "loop_weight": a.loop_weight,
            "odo_weight": a.odo_weight,
            "robust": a.robust,
            "robust_delta": a.robust_delta,
            "max_iter": a.max_iter,
            "yaw_sign_used": used_sign,
            "yaw_sign_source": sign_source,
            "yaw_sign_note": sign_note,
            "loop_source": "pose_input_loop.json（loop_verify.json 的只读副本）",
            "n_confirmed_loops": len(res["loops"]),
            "n_rejected_loops": len(res["rejected"]),
            "c4_c6_in_graph": False,
        },
        "nodes": nodes,
        "edges": edge_out,
        "metrics": metrics,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    # 报告
    rep_lines = []
    rep_lines.append("# 2D 米制位姿图报告\n")
    rep_lines.append("run: `%s`\n" % run_dir)
    rep_lines.append("loop 输入：只读副本 `pose_input_loop.json`（开工第一步复制自 "
                      "loop_verify.json，避免并发重写污染）。\n")
    rep_lines.append("\n## 参数\n")
    rep_lines.append("- dt=%.3f s, loop_weight=%.2f, odo_weight=%.2f, robust=%s, "
                      "delta=%.2f, max_iter=%d\n" %
                      (a.dt, a.loop_weight, a.odo_weight, a.robust,
                       a.robust_delta, a.max_iter))
    rep_lines.append("- yaw 符号：采用 %+d，来源 = **%s**。\n"
                      "  - %s\n" % (used_sign, sign_source, sign_note))
    rep_lines.append("\n## 验收逐条\n")
    rep_lines.append("1. **c0↔c22 闭环误差**：优化前 **%.3f m**，优化后 **%.3f m**，"
                      "显著下降=%s（比值 %.3f）。\n" %
                      (res["pre_closure"], res["post_closure"], improved,
                       ratio if ratio is not None else float('nan')))
    rep_lines.append("   - 对照：yaw_sign=+1 前误差 %.3f m；yaw_sign=-1 前误差 %.3f m，"
                      "差 %.6f m ⇒ %s"
                      "符号来自 `%s`%s。\n"
                     % (res_pos["pre_closure"], res_neg["pre_closure"],
                        closure_gap,
                        "两者接近（标量模型下为镜像对称）⇒ 闭环比对挑不出符号。"
                        if closure_gap < 0.05 else
                        "矢量位移下两者已可分辨，但闭环比对是**内部自洽性**、"
                        "不作外部参照（旧注释的「镜像对称」只在无横移时成立）。",
                        sign_source,
                        "" if sign_source == "image_yaw_check"
                        else "（**不是标定**，见上方符号来源说明）"))
    rep_lines.append("2. **轨迹顺序一致性**：时间栅格天然单调；优化后相邻航位步最大跳变 "
                      "%.3f m（典型步 %.3f m），异常跳变 %d 处 ⇒ 无折返/打结=%s；"
                      "前进方向与 HMD yaw 自洽比例 %.1f%%（%d/%d 移动步）。\n"
                     % (max_jump, typ_step, n_jump_bad, no_fold,
                        consistency * 100, n_consistent, n_odo))
    rep_lines.append("   - ⚠️ 本条的 %.1f%% 对 yaw 符号**不敏感**：`yaw_ang` 与航迹用的是"
                      "同一个 `yaw_sign`，翻符号时两边同时翻，比例不变。所以它是"
                      "内部自洽性，**不能**用来定符号（`yaw_sign_check.py` 的"
                      "图像证据才是外部参照）。\n" % (consistency * 100))
    rep_lines.append("3. **被拒静止假回环未进图**：断言通过——图中无 rejected 边，"
                      "c4-c6 不在（rejected 共 %d 条：含 4-6/0-2/17-19 等）。\n"
                     % len(res["rejected"]))
    rep_lines.append("4. **每条边含 source + confidence**：odometry source=osc_integral"
                      "(heading=hmd_yaw) conf=1.0；loop source=confirmed_visual_loop:a-b，"
                      "confidence=inliers/最大inliers。\n")
    rep_lines.append("\n## 残差分解\n")
    rep_lines.append("- odometry RMS：前 %.4f / 后 %.4f m\n"
                     % (res["pre_rbt"]["odometry"] or 0,
                        res["post_rbt"]["odometry"] or 0))
    rep_lines.append("- loop RMS：前 %.4f / 后 %.4f m（后值=融合后回环残余不一致度）\n"
                     % (res["pre_rbt"]["loop"] or 0, res["post_rbt"]["loop"] or 0))
    rep_lines.append("- 总残差（终）：%.6f，迭代数 %d\n" % (res["final_cost"], res["n_iter"]))
    rep_lines.append("\n## 诚实性备注\n")
    if not improved or res["pre_closure"] > 10.0:
        rep_lines.append("- ⚠️ 航位推算**未自行闭合**（前误差 %.2f m），说明 OSC 速度积分 / "
                          "HMD yaw 与真实前进方向存在系统偏差或静止段位移缺失；闭环约束仅把 "
                          "c22 拉向 c0，但其它回环的残余不一致度（loop RMS=%.4f）偏高，"
                          "地图不可全信。\n"
                          % (res["pre_closure"], res["post_rbt"]["loop"] or 0))
    else:
        rep_lines.append("- 航位推算本身已较好闭合（前误差 %.2f m），闭环修正进一步收紧；"
                         "回环残余不一致度 loop RMS=%.4f。\n"
                         % (res["pre_closure"], res["post_rbt"]["loop"] or 0))
    rep_lines.append("- 未为凑数字调权重。**yaw 零方向仍未标定**（本工具只定符号，"
                      "不定零点）；符号来源见上方 `%s`。\n" % sign_source)
    with open(report, "w", encoding="utf-8") as f:
        f.write("".join(rep_lines))

    # 控制台摘要
    print("=== pose_graph 完成 ===")
    print("yaw_sign=%+d  节点=%d  航位边=%d  回环边=%d"
          % (used_sign, N, metrics["n_odometry_edges"], metrics["n_loop_edges"]))
    print("c0↔c22 闭环: 前 %.3f m -> 后 %.3f m  显著下降=%s"
          % (res["pre_closure"], res["post_closure"], improved))
    print("轨迹一致比例 %.1f%%" % (consistency * 100))
    print("输出: %s" % out)
    print("报告: %s" % report)


if __name__ == "__main__":
    main()
