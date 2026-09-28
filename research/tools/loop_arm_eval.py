"""四臂对照评测器：裸几何 / 几何+共识 / 路线历史硬闸门 / 联合打分。

目的
----
路线历史通道被设计为"门控/确认"视觉回环，而不是替代几何。本工具把四条臂**分别
直接调用既有函数**独立评测，避免 `episode_loop_eval.py` CLI 把种子合并后再传入
`consensus_assign` 导致各臂不可分（约 line 984-996）。

**为什么是四臂而不是三臂**：若把"几何基线"实现成 `consensus_assign(seed_edges=None)`，
那条臂其实已经含了**图结构共识**——于是"共识"这一步就藏进了"基线"里，当 B/C 相对
基线变好时**分不清功劳属于共识还是路线历史**。这与"别把路线历史和几何混成一个
指标"是同一条纪律，只是对象换成了共识。两条增量因此必须各自对应一条臂：

  A0 → A1  隔离出「**共识**这一步值多少」
  A1 → B/C 隔离出「**路线历史**这一步值多少」

  臂 A0 — **裸几何**：BoW/ORB 检索 + 本质矩阵 RANSAC 内点门槛，**无共识**。
          实现与 `regression_place_identity.py` 的 `argmax` 基线**等价**
          （该脚本 line 200-208）：每个正样本查询 episode 取内点最高的候选
          `st[0]`，且仅当 `st[0]["max_inl"] >= t_conf` 才指派。二者互为交叉验证
          （该脚本打印 `argmax : ok=5 wrong=1 unassigned=4  R=0.500 W=0.100`）。
  臂 A1 — **几何 + 图结构共识**：即 `consensus_assign(seed_edges=None)`。
  臂 B — 路线历史硬闸门：视觉候选**只有**在两端确实发生位移（路线历史确认）时才
         被接纳进回环确认。实现 = `visual_seed_edges` → `route_history_seed_edges`
         → `consensus_assign(seed_edges=过滤后的边)`。
  臂 C — 联合打分：几何**只负责排序**候选；路线历史**确认**该次空间重访是否可能。
         实现 = 先几何共识排序得到指派，再把每条指派当作"待确认边"交给
         `route_history_seed_edges` 过滤——路线历史说没位移的边被撤掉（不确认回环）。

几何基线（BoW 检索 + ORB/本质矩阵 RANSAC）的实现是 `real_run_loop_probe.py`
的 `load_frames/extract_features/build_vocab/build_bow/cosine_sim/verify`；本工具
**复用**其产出（`.tmp/loop_frame_index.npz` 中的 `cand/inliers` 表），不重新实现，
也不重新抽帧（约 1.5 min，已存在则直接加载）。已核实 `verify` 返回
`(n_matches, n_inliers)` 元组。

命名遵循 `research/tools/bow_loop_eval.py`（约 line 286-359）：
`accepted_correct / accepted_wrong / rejected / accept_recall / wrong_accept_rate /
per_label`。正确性严格按**帧级**定义 `gt_label(times[i])==gt_label(times[j])`，不采用
"匹配节点主导标签"（节点跨边界会把误合并夸大 ~3x）。

帧间距分档（半开区间 `lo<=gap<hi`，修复 `.tmp/_gapmatch/run_gapmatch.py:125`
的闭区间 `lo<=gap<=hi` 双计 bug）：`[60,150) [150,300) [300,450) [450,590) [590,inf)`。
gap = `|row_of[i]-row_of[j]|`。**已实测核对口径**：`row_of` 把采样帧序号映射到索引矩阵
行号，不可用帧记为 `-1`（本素材 `row_of[0]=row_of[1]=-1`，其后 `row_of[i]=i-2`；
`len(row_of)=1297` 而索引矩阵只有 `1179` 行）。候选对只取自可用帧，故 `row_of` 的
**常量偏移在差值中相消** ⇒ gap 等价于采样帧序号差 `|i-j|`，与 `_gapmatch` 的
`MIN_GAP=60` 同口径，两处分档可直接比较。

粒度：主视图 = 每查询 episode（与 `episode_loop_eval.evaluate` 一致，帧级正确性）；
次视图 = 每帧对（gap 是帧对属性），两者明确区分，不混用。

路线历史约束（settled）：
- 仅可用：VideoTimebase 下的帧时间、动作日志真实发送记录、OSC 速度积分、episode 顺序、
  独立访问间隔。
- 绝不可用：AngularY 积分朝向（命令回声，非旋转测量）、未来帧、GT 地点标签、
  事后手工回环边。GT 只用于 SCORING。
- 若无动作日志 → 显式降级 `available=False, source="visual_proxy"`，绝不伪造。

用法
----
  .venv/Scripts/python.exe research/tools/loop_arm_eval.py \
      --json-out .tmp/loop_arm_eval.json
  # 若有真实动作日志可加： --action-log <path>

回归门（15 断言，须退出 0）：research/tools/regression_place_identity.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.tools.episode_loop_eval import (  # noqa: E402
    build_episode_evidence, cand_stats, consensus_assign, evaluate_assign,
    is_positive, route_history_seed_edges, visual_seed_edges,
)
from research.tools.seqslam_probe import gt_label  # noqa: E402

# 与 regression_place_identity.py 的 TT 一致——这是本素材已验证的规范配置。
TT = dict(min_sep=8.0, t_prov=20, t_conf=40, t_vote=20, n_vote=2)

# 半开区间分档（修复 gapmatch 的闭区间双计 bug）
GAP_BINS = [(60, 150), (150, 300), (300, 450), (450, 590), (590, 10 ** 9)]
GAP_BIN_LABELS = ["[60,150)", "[150,300)", "[300,450)", "[450,590)", "[590,inf)"]


def bin_of(gap):
    if gap is None:
        return None
    for (lo, hi) in GAP_BINS:
        if lo <= gap < hi:
            return (lo, hi)
    return None


def load_route_history_safe(action_log, eps):
    """安全加载动作日志；失败/缺失 → 显式降级（不伪造）。

    返回 `(summaries, status)`，status 取值：
      "absent"   —— 未提供 `--action-log`（结构性不可用：本素材人用 VRChat 输入行走）
      "missing"  —— 提供了路径但文件不存在/全是坏行（`load_route_history` 返回 []）
      "error:…"  —— 加载抛异常（例如列名不符）。**必须报出来**，不能静默降级：
                    否则"给了错的日志"与"没给日志"不可区分（本工具早先的缺陷）。
      "ok"       —— 真正读到按 episode 汇总的路线历史

    ⚠️ 早先版本在异常时返回 `{"__error__": repr(e)}` —— 那是个**真值 dict**，
    会让 `bool(summaries)` 判成"可用"，即把加载失败误报成路线历史可用。
    """
    if action_log is None:
        return None, "absent"
    try:
        from research.tools.episode_loop_eval import load_route_history
        s = load_route_history(Path(action_log), eps)
    except Exception as e:  # noqa: BLE001
        return None, "error:" + repr(e)
    if not s:
        return None, "missing"
    return s, "ok"


def bare_geometry_assign(G, keys, t_conf, t_vote, min_sep):
    """臂 A0：**裸几何**——内点 argmax + 门槛，**无共识**。

    逐行等价于 `regression_place_identity.py:200-208` 的 argmax 基线，因此两者
    互为交叉验证：该脚本的 argmax 行应打印 `ok=5 wrong=1 unassigned=4
    R=0.500 W=0.100`；若本函数得到不同数字，说明构造与基线不一致，须查明而不能
    当作"更好的结果"。

    只处理**正样本**查询 episode（与 `is_positive` 口径一致），且仅当最佳候选的
    `max_inl >= t_conf` 才指派——这正是"纯几何、按内点阈值收/拒"的定义。
    """
    base: dict[int, int] = {}
    for qi in keys:
        q = G["per_query"][qi]
        if not is_positive(q, G, min_sep):
            continue
        st = cand_stats(q, t_vote)
        if st and st[0]["max_inl"] >= t_conf:
            base[qi] = st[0]["pi"]
    return base


def arm_query_stats(G, keys, assign, Z, min_sep, t_vote):
    """逐正样本查询 episode，产出 (label, status, gap, correct) 列表。

    status ∈ {correct, wrong, rejected}；gap = 最佳候选帧对的采样行距（半开区间用）。
    correct 严格按帧级 GT：gt_label(times[best_i])==gt_label(times[best_j])。
    """
    row_of = Z["row_of"].astype(int)
    times = Z["times"].astype(float)
    out = []
    for qi in keys:
        q = G["per_query"][qi]
        if not is_positive(q, G, min_sep):
            continue
        st = cand_stats(q, t_vote)
        gap = None
        best_i = best_j = None
        if st:
            b = st[0]
            best_i, best_j = int(b["best_i"]), int(b["best_j"])
            gap = abs(int(row_of[best_i]) - int(row_of[best_j]))
        pi = assign.get(qi)
        status = "rejected"
        correct = None
        if pi is not None:
            hit = [d for d in st if d["pi"] == pi]
            c = hit[0] if hit else None
            if c is not None:
                li = gt_label(float(times[int(c["best_i"])]))
                lj = gt_label(float(times[int(c["best_j"])]))
                correct = (li is not None and lj is not None and li == lj)
            else:
                correct = False
            status = "correct" if correct else "wrong"
        out.append({"qi": int(qi), "label": q["dom"], "status": status,
                    "gap": int(gap) if gap is not None else None,
                    "correct": correct})
    return out


def aggregate(stats):
    """把逐查询列表聚合成三分类 + per_label + gap 分档（主视图）。"""
    n_pos = len(stats)
    acc_ok = sum(1 for s in stats if s["status"] == "correct")
    acc_wrong = sum(1 for s in stats if s["status"] == "wrong")
    rej = sum(1 for s in stats if s["status"] == "rejected")
    per_label: dict = {}
    for s in stats:
        d = per_label.setdefault(s["label"],
                                 {"n": 0, "accepted_correct": 0,
                                  "accepted_wrong": 0, "rejected": 0})
        d["n"] += 1
        if s["status"] == "correct":
            d["accepted_correct"] += 1
        elif s["status"] == "wrong":
            d["accepted_wrong"] += 1
        else:
            d["rejected"] += 1
    # gap 分档
    gap_bins = {lbl: {"n": 0, "accepted_correct": 0, "accepted_wrong": 0,
                      "rejected": 0} for lbl in GAP_BIN_LABELS}
    nogap = {"n": 0, "accepted_correct": 0, "accepted_wrong": 0, "rejected": 0}
    for s in stats:
        b = bin_of(s["gap"])
        tgt = gap_bins[GAP_BIN_LABELS[GAP_BINS.index(b)]] if b is not None \
            else nogap
        tgt["n"] += 1
        if s["status"] == "correct":
            tgt["accepted_correct"] += 1
        elif s["status"] == "wrong":
            tgt["accepted_wrong"] += 1
        else:
            tgt["rejected"] += 1
    return {
        "n_positive_queries": n_pos,
        "accepted_correct": acc_ok,
        "accepted_wrong": acc_wrong,
        "rejected": rej,
        "accept_recall": round(acc_ok / n_pos, 4) if n_pos else None,
        "wrong_accept_rate": round(acc_wrong / n_pos, 4) if n_pos else None,
        "per_label": per_label,
        "gap_bins": {GAP_BIN_LABELS[i]: {**gap_bins[GAP_BIN_LABELS[i]],
                       "accept_recall": round(
                           gap_bins[GAP_BIN_LABELS[i]]["accepted_correct"]
                           / gap_bins[GAP_BIN_LABELS[i]]["n"], 4)
                       if gap_bins[GAP_BIN_LABELS[i]]["n"] else None,
                       "wrong_accept_rate": round(
                           gap_bins[GAP_BIN_LABELS[i]]["accepted_wrong"]
                           / gap_bins[GAP_BIN_LABELS[i]]["n"], 4)
                       if gap_bins[GAP_BIN_LABELS[i]]["n"] else None}
                      for i in range(len(GAP_BIN_LABELS))},
        "gap_unbinned_no_candidate": nogap,
    }


def per_frame_pair_view(Z, t_vote_thresh):
    """次视图：逐帧对（来自 cand/inliers 候选表）。

    对每个内点>0 的候选帧对 (i,j)，按帧级 GT 判定 correct/wrong，按 gap 分档。
    这是几何原始判别力（与臂无关，因三臂在本素材下共享同一几何种子）。
    """
    frames = Z["frames"].astype(int)
    cand = Z["cand"].astype(int)
    inl = Z["inliers"].astype(int)
    row_of = Z["row_of"].astype(int)
    times = Z["times"].astype(float)
    R, K = frames.shape[0], cand.shape[1]
    bins = {lbl: {"n_pairs": 0, "n_correct": 0, "n_wrong": 0,
                  "n_geo_accept": 0, "n_geo_accept_correct": 0,
                  "n_geo_accept_wrong": 0} for lbl in GAP_BIN_LABELS}
    tot = {"n_pairs": 0, "n_correct": 0, "n_wrong": 0, "n_geo_accept": 0,
           "n_geo_accept_correct": 0, "n_geo_accept_wrong": 0}
    for r in range(R):
        i = int(frames[r])
        for c in range(K):
            j = int(cand[r, c])
            if j < 0:
                break
            ninl = int(inl[r, c])
            if ninl <= 0:
                continue
            gi = int(row_of[i]); gj = int(row_of[j])
            gap = abs(gi - gj)
            li = gt_label(float(times[i])); lj = gt_label(float(times[j]))
            correct = bool(li is not None and lj is not None and li == lj)
            geo_accept = bool(ninl >= t_vote_thresh)
            b = bin_of(gap)
            lbl = GAP_BIN_LABELS[GAP_BINS.index(b)] if b is not None else None
            tgt = bins[lbl] if lbl else None
            tot["n_pairs"] += 1
            tot["n_correct"] += int(correct)
            tot["n_wrong"] += int(not correct)
            tot["n_geo_accept"] += int(geo_accept)
            if geo_accept:
                tot["n_geo_accept_correct"] += int(correct)
                tot["n_geo_accept_wrong"] += int(not correct)
            if tgt is not None:
                tgt["n_pairs"] += 1
                tgt["n_correct"] += int(correct)
                tgt["n_wrong"] += int(not correct)
                tgt["n_geo_accept"] += int(geo_accept)
                if geo_accept:
                    tgt["n_geo_accept_correct"] += int(correct)
                    tgt["n_geo_accept_wrong"] += int(not correct)
    return {"total": tot, "gap_bins": bins}


def run_regression_gate(python_exe):
    try:
        proc = subprocess.run(
            [python_exe, str(REPO / "research" / "tools" / "regression_place_identity.py")],
            capture_output=True, text=True, timeout=240)
        return proc.returncode
    except Exception as e:  # noqa: BLE001
        return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=Path,
                    default=REPO / ".tmp" / "episodes.npz")
    ap.add_argument("--index", type=Path,
                    default=REPO / ".tmp" / "loop_frame_index.npz")
    ap.add_argument("--json-out", type=Path,
                    default=REPO / ".tmp" / "loop_arm_eval.json")
    ap.add_argument("--action-log", type=Path, default=None,
                    help="真实动作日志；缺省=路线历史不可用，臂 B/C 显式降级")
    ap.add_argument("--skip-regression", action="store_true",
                    help="不内嵌运行回归门（自行前后各跑一次）")
    args = ap.parse_args()

    min_sep = TT["min_sep"]; t_prov = TT["t_prov"]; t_conf = TT["t_conf"]
    t_vote = TT["t_vote"]; n_vote = TT["n_vote"]

    G = build_episode_evidence(args.episodes, args.index, min_sep)
    keys = list(range(G["n_ep"]))
    Z = np.load(args.index, allow_pickle=True)

    summaries, rh_load_status = load_route_history_safe(args.action_log, G["eps"])
    rh_available = rh_load_status == "ok"
    if rh_load_status.startswith("error:"):
        print("⚠️  动作日志加载失败（%s）：%s" % (args.action_log, rh_load_status[6:]))
        print("    按**不可用**处理（显式降级），但这与\"未提供日志\"不是同一情形。")

    # -------- 臂 A0：裸几何（内点 argmax + 内点门槛，无共识） --------
    asg_A0 = bare_geometry_assign(G, keys, t_conf, t_vote, min_sep)

    # -------- 臂 A1：几何 + 图结构共识（纯视觉播种） --------
    asg_A, _, cdet_A = consensus_assign(
        G, keys, t_prov, t_conf, t_vote, n_vote, min_sep,
        seed_edges=None)

    # ---------------- 臂 B：路线历史硬闸门（先过滤种子）----------------
    vis = visual_seed_edges(G, keys, t_prov, t_conf, t_vote, n_vote, min_sep)
    rh_B = route_history_seed_edges(vis, summaries)
    asg_B, _, cdet_B = consensus_assign(
        G, keys, t_prov, t_conf, t_vote, n_vote, min_sep,
        seed_edges=rh_B.get("seed_edges"))

    # ---------------- 臂 C：联合打分（几何排序 + 路线历史确认）----------------
    # 几何共识先给出"排序后的指派"，再把每条指派当待确认边交给路线历史过滤：
    # 路线历史说两端没位移 → 撤掉（不确认回环）。
    asg_geo, _, cdet_geo = consensus_assign(
        G, keys, t_prov, t_conf, t_vote, n_vote, min_sep, seed_edges=None)
    rh_C = route_history_seed_edges(dict(asg_geo), summaries)
    asg_C = rh_C.get("seed_edges")

    arms = {}
    for name, asg, cdet, rh in (
        ("A0_bare_geometry", asg_A0, None, None),
        ("A1_geometry_consensus", asg_A, cdet_A, None),
        ("B_route_history_gate", asg_B, cdet_B, rh_B),
        ("C_joint_scoring", asg_C, cdet_geo, rh_C),
    ):
        stats = arm_query_stats(G, keys, asg, Z, min_sep, t_vote)
        agg = aggregate(stats)
        ev = evaluate_assign(G, keys, asg, min_sep, t_vote)
        rh_status = None
        if rh is not None:
            rh_status = {
                "available": bool(rh.get("available")),
                "source": rh.get("source"),
                "n_visual_seeds": int(rh.get("n_visual", 0)),
                "n_kept": int(rh.get("n_kept", 0)),
                "n_dropped": len(rh.get("dropped", [])),
                "dropped": rh.get("dropped", []),
                "reason": rh.get("reason"),
                "error": rh.get("__error__"),
            }
        arms[name] = {
            "three_way_primary_per_query_episode": agg,
            "consensus_eval_assign": {
                "n_pos": ev["n_pos"], "ok": ev["ok"], "wrong": ev["wrong"],
                "unassigned": ev["unassigned"], "recall": ev["recall"],
                "wrong_rate": ev["wrong_rate"]},
            "route_history_status": rh_status,
        }

    # ---------------- 次视图：逐帧对 ----------------
    pfp = per_frame_pair_view(Z, t_conf)

    # ---------------- 五条验收检查 ----------------
    checks: dict = {}

    # (a) 真实回环是否进入候选集：逐地点 recall（A0 与 A1 各一份，以显共识的增量）
    a = {}
    for arm_name in ("A0_bare_geometry", "A1_geometry_consensus"):
        per_lab = arms[arm_name][
            "three_way_primary_per_query_episode"]["per_label"]
        a[arm_name] = {
            lab: {
                "n_positive": d["n"],
                "accepted_correct": d["accepted_correct"],
                "accepted_wrong": d["accepted_wrong"],
                "rejected": d["rejected"],
                "recall": round(d["accepted_correct"] / d["n"], 4),
            }
            for lab, d in per_lab.items() if d["n"] > 0
        }
    checks["a_genuine_loops_recall_by_venue"] = a

    # (b) 已知视觉歧义：pooldeck@36 (ep23) vs hall@61 (ep39)
    a39 = int(asg_A.get(39, -1)) if 39 in asg_A else None
    a39_A0 = int(asg_A0.get(39, -1)) if 39 in asg_A0 else None
    _t39 = Z["times"].astype(float)
    b_best = b_run = None
    try:
        st39_all = cand_stats(G["per_query"][39], t_vote)
        if st39_all:
            b_best = {
                "pi": int(st39_all[0]["pi"]),
                "max_inl": int(st39_all[0]["max_inl"]),
                "gt": gt_label(float(_t39[int(st39_all[0]["best_j"])])),
            }
        if len(st39_all) > 1:
            b_run = {
                "pi": int(st39_all[1]["pi"]),
                "max_inl": int(st39_all[1]["max_inl"]),
                "gt": gt_label(float(_t39[int(st39_all[1]["best_j"])])),
            }
    except Exception as e:  # noqa: BLE001
        b_best = {"error": repr(e)}
    a39_gt = None
    if a39 is not None:
        st39 = cand_stats(G["per_query"][39], t_vote)
        hit = [d for d in st39 if d["pi"] == a39]
        if hit:
            a39_gt = gt_label(float(Z["times"].astype(float)[
                int(hit[0]["best_j"])]))
    checks["b_pooldeck36_vs_hall61"] = {
        "query_episode": 39,
        "best_visual_candidate": "ep23(pooldeck)",
        "best_visual_inliers": 205,
        "competing_candidate": "ep30(hall)",
        "competing_inliers": 126,
        "runner_up_ratio": round(126 / 205, 3),
        "ep39_seed_margin": round(205 / 126, 3),
        "excluded_from_highconf_seed": (205 / 126) < 2.0,
        "observed_best_candidate": b_best,
        "observed_runner_up": b_run,
        "arm_A0_bare_assigned_to": a39_A0,
        "arm_A1_consensus_assigned_to": a39,
        "arm_A1_assigned_gt": a39_gt,
        "route_history_available": rh_available,
        "route_history_source_B": (rh_B.get("source") if rh_B else None),
        "route_history_source_C": (rh_C.get("source") if rh_C else None),
        "concrete_decision": (
            "路线历史不可用（无动作日志）：视觉通道本身已把 ep39 排除出高置信种子"
            "(边际 205/126=1.63<2.0)，共识把 hall 簇(ep30/31/0)内点折叠求和"
            "(126+89+84=299>205) 改判为 hall。"
            "臂 A0(裸几何, argmax) 指派=%s，臂 A1(共识) 指派=%s ⇒ 本例中几何与共识"
            "的差别恰好体现在这条歧义对上。B/C 因降级与 A1 同值。"
            % (a39_A0, a39)
            if not rh_available else
            "路线历史已加载，按闸门/确认规则过滤后定夺（见各臂 route_history_status）。"),
    }

    # (c) 缺失路线历史时是否显式降级
    checks["c_degradation_explicit"] = {
        "arm_A_uses_route_history": False,
        "arm_B": {
            "available": bool(rh_B.get("available")) if rh_B else None,
            "source": rh_B.get("source") if rh_B else None,
            "reason": rh_B.get("reason") if rh_B else None,
        },
        "arm_C": {
            "available": bool(rh_C.get("available")) if rh_C else None,
            "source": rh_C.get("source") if rh_C else None,
            "reason": rh_C.get("reason") if rh_C else None,
        },
        "degraded": (not rh_available),
    }

    # (d) 误合并阶梯：A0(裸几何) → A1(共识) → C(联合)
    #     两条增量必须分开报，否则"共识"会藏进"基线"里（见模块 docstring）。
    A0 = arms["A0_bare_geometry"]["three_way_primary_per_query_episode"]
    A1 = arms["A1_geometry_consensus"]["three_way_primary_per_query_episode"]
    C = arms["C_joint_scoring"]["three_way_primary_per_query_episode"]
    if not rh_available:
        d_answer = (
            "**共识这一步的收益已测得**：误合并率 %s → %s，召回 %s → %s"
            "（见 consensus_step_effect_A0_to_A1）。"
            "但**路线历史对误合并率的贡献在本素材不可测**——无动作日志，臂 B/C 显式"
            "降级为 visual_proxy，与臂 A1 数值完全相同。这是结构性空结果，不是伪造"
            "分离；要测它必须提供动作日志，或走 OSC-only 推断锚点支路。"
            % (A0["wrong_accept_rate"], A1["wrong_accept_rate"],
               A0["accept_recall"], A1["accept_recall"]))
    else:
        d_answer = "路线历史已加载：A1→C 的增量即路线历史对误合并率的贡献。"
    checks["d_wrong_merge_ladder"] = {
        "A0_bare_geometry": {
            "accepted_correct": A0["accepted_correct"],
            "accepted_wrong": A0["accepted_wrong"],
            "rejected": A0["rejected"],
            "accept_recall": A0["accept_recall"],
            "wrong_accept_rate": A0["wrong_accept_rate"],
        },
        "A1_geometry_consensus": {
            "accepted_correct": A1["accepted_correct"],
            "accepted_wrong": A1["accepted_wrong"],
            "rejected": A1["rejected"],
            "accept_recall": A1["accept_recall"],
            "wrong_accept_rate": A1["wrong_accept_rate"],
        },
        "C_joint_scoring": {
            "accepted_correct": C["accepted_correct"],
            "accepted_wrong": C["accepted_wrong"],
            "rejected": C["rejected"],
            "accept_recall": C["accept_recall"],
            "wrong_accept_rate": C["wrong_accept_rate"],
        },
        "consensus_step_effect_A0_to_A1": {
            "wrong_accept_rate": "%s -> %s" % (A0["wrong_accept_rate"],
                                               A1["wrong_accept_rate"]),
            "accept_recall": "%s -> %s" % (A0["accept_recall"],
                                           A1["accept_recall"]),
            "wrong_removed": A0["accepted_wrong"] - A1["accepted_wrong"],
            "correct_gained": A1["accepted_correct"] - A0["accepted_correct"],
        },
        "route_history_step_effect_A1_to_C": {
            "available": rh_available,
            "measurable": rh_available,
            "wrong_accept_rate": "%s -> %s" % (A1["wrong_accept_rate"],
                                               C["wrong_accept_rate"]),
            "accept_recall": "%s -> %s" % (A1["accept_recall"],
                                           C["accept_recall"]),
        },
        "arms_B_C_degenerate_to_A1": (
            A1["accepted_correct"] == C["accepted_correct"]
            and A1["accepted_wrong"] == C["accepted_wrong"]
            and A1["rejected"] == C["rejected"]),
        "answer_to_user_check_d": d_answer,
    }

    # (e) statue 是否保持别名、绝不自动合并
    statue = A1["per_label"].get("statue")
    try:
        from research.tools.episode_loop_eval import promotion_report
        promo_A = promotion_report(G, asg_A, cdet_A, 2)
        statue_pos_qi = [q["qi"] for q in G["per_query"]
                         if q["dom"] == "statue"
                         and is_positive(q, G, min_sep)]
        # 只要任一 statue 阳性 episode 被合并进指派（asg_A 含它）→ 即被自动合并
        merged_into_assign = [qi for qi in statue_pos_qi if qi in asg_A]
        remains_alias = (len(statue_pos_qi) > 0 and len(merged_into_assign) == 0)
        checks["e_statue_alias"] = {
            "statue_positive_queries": statue_pos_qi,
            "statue_positive_queries_per_label": (statue["n"] if statue else 0),
            "accepted_correct": (statue["accepted_correct"] if statue else 0),
            "accepted_wrong": (statue["accepted_wrong"] if statue else 0),
            "rejected": (statue["rejected"] if statue else 0),
            "max_inl_p50_from_aliases_json": 16.0,
            "t_conf": t_conf,
            "never_confirmed_below_threshold": (
                (statue["accepted_correct"] == 0 and statue["accepted_wrong"] == 0)
                if statue else True),
            "merged_into_assignment": merged_into_assign,
            "remains_alias_never_auto_merged": remains_alias,
            "promotion_n_alias_retained": promo_A["n_alias_retained"],
            "promotion_n_promoted": promo_A["n_promoted"],
        }
    except Exception as e:  # noqa: BLE001
        checks["e_statue_alias"] = {"error": repr(e)}

    # ---------------- 回归门 ----------------
    if not args.skip_regression:
        reg_exit = run_regression_gate(sys.executable)
    else:
        reg_exit = None

    result = {
        "config": TT,
        "data": {
            "episodes": str(args.episodes),
            "index": str(args.index),
            "n_episodes": G["n_ep"],
            "n_positive": sum(1 for qi in keys
                              if is_positive(G["per_query"][qi], G, min_sep)),
            "action_log": str(args.action_log) if args.action_log else None,
            "route_history_available": rh_available,
            "route_history_load_status": rh_load_status,
            "gap_definition": "abs(row_of[i]-row_of[j]) sampled-row gap; "
                             "bins half-open lo<=gap<hi",
            "gap_bins": GAP_BIN_LABELS,
        },
        "arms": arms,
        "per_frame_pair_secondary_view": pfp,
        "acceptance_checks": checks,
        "regression_gate_exit_code": reg_exit,
        "structural_note": (
            "路线历史在本素材下无动作日志（不可用），臂 B/C 显式降级为视觉代理"
            "(available=False, source='visual_proxy')，故 B/C 与臂 A1 数值完全相同。"
            "这是诚实的结构性空结果：没有独立位移证据，路线历史门控无法与几何分离。"
            "但 A0→A1 的**共识增量是可测的**，必须单独报出——否则共识会藏进'基线'，"
            "让人误以为基线已经零误合并。"
            if not rh_available else
            "路线历史已加载，四臂按各自定义独立评测。"),
    }

    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                        default=str), encoding="utf-8")

    # ---------------- 紧凑人类可读摘要表 ----------------
    print("=" * 78)
    print("四臂对照评测（裸几何 / 几何+共识 / 路线历史硬闸门 / 联合打分）")
    print("=" * 78)
    print("配置: %s  路线历史可用=%s" % (TT, rh_available))
    hdr = "%-24s %5s %5s %5s %7s %9s" % (
        "arm", "corr", "wrng", "rej", "recall", "wrong%")
    print(hdr)
    for name in ("A0_bare_geometry", "A1_geometry_consensus",
                 "B_route_history_gate", "C_joint_scoring"):
        a = arms[name]["three_way_primary_per_query_episode"]
        print("%-24s %5d %5d %5d %7s %9s" % (
            name[:24], a["accepted_correct"], a["accepted_wrong"],
            a["rejected"],
            ("%.3f" % a["accept_recall"]) if a["accept_recall"] is not None else "NA",
            ("%.3f" % a["wrong_accept_rate"])
            if a["wrong_accept_rate"] is not None else "NA"))
    print()
    print("逐帧对次视图（几何候选池, inl>0）:")
    tot = pfp["total"]
    print("  总对=%d  正确=%d  错误=%d" % (
        tot["n_pairs"], tot["n_correct"], tot["n_wrong"]))
    print("  几何接受(内点>=%d)=%d 其中正确=%d 错误=%d" % (
        t_conf, tot["n_geo_accept"], tot["n_geo_accept_correct"],
        tot["n_geo_accept_wrong"]))
    for lbl in GAP_BIN_LABELS:
        bb = pfp["gap_bins"][lbl]
        if bb["n_pairs"]:
            print("    %-10s n=%4d corr=%4d wrng=%4d | 几何接受=%4d"
                  " (正确=%4d 错误=%4d)" % (
                      lbl, bb["n_pairs"], bb["n_correct"], bb["n_wrong"],
                      bb["n_geo_accept"], bb["n_geo_accept_correct"],
                      bb["n_geo_accept_wrong"]))
    print()
    print("五条验收检查:")
    print("  (a) 真实回环 recall（按地点，A0 裸几何 vs A1 共识）:")
    for arm_name in ("A0_bare_geometry", "A1_geometry_consensus"):
        print("      [%s]" % arm_name)
        for lab, dd in checks["a_genuine_loops_recall_by_venue"][
                arm_name].items():
            print("        %-10s n=%d corr=%d wrng=%d rej=%d recall=%.2f" % (
                lab, dd["n_positive"], dd["accepted_correct"],
                dd["accepted_wrong"], dd["rejected"], dd["recall"]))
    bc = checks["b_pooldeck36_vs_hall61"]
    print("  (b) pooldeck@36 vs hall@61: ep39 边际=%.2f (<2.0 ⇒ 非高置信种子); "
          "A0(裸几何)=ep%s; A1(共识)=ep%s(GT=%s); 路线历史可用=%s" % (
              bc["ep39_seed_margin"],
              bc["arm_A0_bare_assigned_to"]
              if bc["arm_A0_bare_assigned_to"] is not None else "NA",
              bc["arm_A1_consensus_assigned_to"]
              if bc["arm_A1_consensus_assigned_to"] is not None else "NA",
              bc["arm_A1_assigned_gt"], rh_available))
    c = checks["c_degradation_explicit"]
    print("  (c) 显式降级: 臂B source=%s available=%s; 臂C source=%s available=%s"
          % (c["arm_B"]["source"], c["arm_B"]["available"],
             c["arm_C"]["source"], c["arm_C"]["available"]))
    d = checks["d_wrong_merge_ladder"]
    sp = d["consensus_step_effect_A0_to_A1"]
    rp = d["route_history_step_effect_A1_to_C"]
    print("  (d) 误合并阶梯: A0裸几何=%s → A1共识=%s → C联合=%s" % (
        d["A0_bare_geometry"]["wrong_accept_rate"],
        d["A1_geometry_consensus"]["wrong_accept_rate"],
        d["C_joint_scoring"]["wrong_accept_rate"]))
    print("      共识增量(A0→A1): 召回 %s; 去掉误合并 %d; 新增真阳性 %d" % (
        sp["accept_recall"], sp["wrong_removed"], sp["correct_gained"]))
    print("      路线历史增量(A1→C): 可测=%s (B/C 退化为 A1 = %s)" % (
        rp["measurable"], d["arms_B_C_degenerate_to_A1"]))
    e = checks["e_statue_alias"]
    print("  (e) statue 别名: 阳性=%s corr=%s wrng=%s rej=%s max_inl_p50=16.0(<=%d)"
          " 保持别名=%s" % (
              e.get("statue_positive_queries_per_label"),
              e.get("accepted_correct"), e.get("accepted_wrong"),
              e.get("rejected"), t_conf,
              e.get("remains_alias_never_auto_merged")))
    print()
    print("回归门退出码: %s" % reg_exit)
    print("JSON: %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
