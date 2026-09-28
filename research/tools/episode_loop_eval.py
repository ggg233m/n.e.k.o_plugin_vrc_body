"""episode 级三级确认协议评测：PlaceEpisode -> LoopCandidate -> PlaceIdentity。

与 `loop_place_vote_eval.py` 的区别
----------------------------------
`loop_place_vote_eval.py` 的"地点"要么是真值段（有泄漏）要么是时间邻近聚类（启发式）。
本脚本的"地点"是 **`episode_segment.py` 在线切分出来的 PlaceEpisode**——
切分只用 BoW 相似度 + 运动阶段 + 场景结构，不用真值、不用未来帧、按秒表达门限。

投票是**地点级、偏移无关**的：对每个查询 episode Q，对每个更早的 episode P，
统计 Q 的帧各自独立匹配到 P 的证据，取最长连续支持长度。
禁止对角形式 `(i-lag, j-lag)`。

三级语义（用户 2026-09-20 定稿）
--------------------------------
- ``candidate``       : BoW Top-K 命中，只表示"值得检查"
- ``provisional_loop``: lag0 ORB 证据达低门槛；允许写入候选回环记录，**不合并地点身份**
- ``confirmed_loop``  : 多帧连续支持 + 证据强度 + 无重复纹理风险；才允许合并 PlaceIdentity

评测口径
--------
- 逐地点报告：episode 数、provisional 召回、confirmed 召回、错误合并、被拒绝数
- 不把"全部拒绝"计作成功：错误率为 0 但召回为 0 的配置会被显式标注
- leave-one-loop-out：留出地点上重新选参，报告留出成绩

用法
----
  .venv/Scripts/python.exe research/tools/episode_loop_eval.py \
      --episodes .tmp/episodes.npz --index .tmp/loop_frame_index.npz \
      --json-out .tmp/episode_loop_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ------------------------------------------------------------------ 证据表

def build_episode_evidence(ep_path: Path, idx_path: Path, min_sep: float,
                           eps_override=None):
    """返回每个查询 episode 对所有更早 episode 的证据。"""
    E = np.load(ep_path, allow_pickle=True)
    eps = (np.asarray(eps_override, dtype=np.int64) if eps_override is not None
           else E["episodes"].astype(np.int64))     # (n_ep, 2) 帧区间 [a, b)
    times_all = E["times"].astype(np.float64)
    lab_all = E["label"].astype(str)

    Z = np.load(idx_path, allow_pickle=True)
    frames = Z["frames"].astype(np.int64)
    row_of = Z["row_of"].astype(np.int64)
    CAND = Z["cand"].astype(np.int64)
    INL = Z["inliers"].astype(np.int64)
    CEL = Z["cells"].astype(np.int64)
    REM = Z["remaining"].astype(np.int64)
    times = Z["times"].astype(np.float64)
    label = Z["label"].astype(str)
    K = int(Z["topk"])

    # 每个 episode 的帧（原始帧号，按时间序）与其 dominant label
    ep_frames: list[list[int]] = []
    ep_dom: list[str | None] = []
    ep_t: list[tuple[float, float]] = []
    for a, b in eps:
        fs = [int(i) for i in range(int(a), int(b)) if row_of[int(i)] >= 0]
        ep_frames.append(fs)
        cnt: dict[str, int] = {}
        for i in fs:
            L = label[i]
            if L and L != "None":
                cnt[L] = cnt.get(L, 0) + 1
        ep_dom.append(max(cnt, key=cnt.get) if cnt else None)
        ep_t.append((float(times[fs[0]]), float(times[fs[-1]])) if fs else (0.0, 0.0))

    n_ep = len(eps)
    # 每个 episode 的候选 episode 集合（更早，且间隔 >= min_sep 秒）
    per_query = []
    for qi in range(n_ep):
        qs = ep_t[qi][0]
        earlier = [pi for pi in range(n_ep)
                   if pi != qi and ep_t[pi][1] + min_sep <= qs and ep_frames[pi]]
        qfs = ep_frames[qi]
        ev = {}
        for pi in earlier:
            pset = set(ep_frames[pi])
            # 逐帧独立证据：该帧的 Top-K 是否落在 P 内
            hit = np.zeros(len(qfs), dtype=bool)      # BoW 命中
            good = np.zeros(len(qfs), dtype=bool)     # 几何通过（inliers >= t_vote 用表存原始值）
            best_inl = np.zeros(len(qfs), dtype=np.int64)
            best_cel = np.zeros(len(qfs), dtype=np.int64)
            best_rem = np.zeros(len(qfs), dtype=np.int64)
            best_j = np.full(len(qfs), -1, dtype=np.int64)
            for k, i in enumerate(qfs):
                r = row_of[i]
                for c in range(K):
                    j = int(CAND[r, c])
                    if j < 0:
                        break
                    if j in pset:
                        hit[k] = True
                        if int(INL[r, c]) > best_inl[k]:
                            best_inl[k] = int(INL[r, c])
                            best_cel[k] = int(CEL[r, c])
                            best_rem[k] = int(REM[r, c])
                            best_j[k] = j
                        break
            good = best_inl > 0
            ev[pi] = {"hit": hit, "inl": best_inl, "cel": best_cel,
                      "rem": best_rem, "j": best_j, "good": good}
        per_query.append({"qi": qi, "frames": qfs, "dom": ep_dom[qi],
                          "t0": ep_t[qi][0], "t1": ep_t[qi][1],
                          "earlier": earlier, "ev": ev})

    return {"n_ep": n_ep, "eps": eps, "ep_frames": ep_frames, "ep_dom": ep_dom,
            "ep_t": ep_t, "per_query": per_query, "label": label,
            "times": times, "K": K, "min_sep": min_sep}


# ------------------------------------------------------------------ 判定

def longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def cand_stats(q, t_vote: int):
    """该查询 episode 对**全部**更早 episode 的证据，按强度排序。

    用于"别名状态"：不再只留 argmax，而是保留全部竞争假设。
    """
    out = []
    for pi, e in q["ev"].items():
        inl = e["inl"]
        if inl.size == 0:
            continue
        mx = int(inl.max())
        if mx <= 0:
            continue
        k = int(np.argmax(inl))
        if e["j"][k] < 0:
            continue
        out.append({"pi": pi, "max_inl": mx,
                    "run": longest_run(inl >= t_vote) if t_vote > 0 else int(inl.size),
                    "npairs": int((inl >= t_vote).sum()) if t_vote > 0 else int(inl.size),
                    "cells": int(e["cel"][k]), "rem": int(e["rem"][k]),
                    "best_i": int(q["frames"][k]), "best_j": int(e["j"][k])})
    out.sort(key=lambda d: (-d["max_inl"], -d["run"], d["pi"]))
    return out


def pick_candidate(q, t_vote: int):
    """在查询 episode 的所有更早 episode 里选证据最强的一个。"""
    st = cand_stats(q, t_vote)
    if not st:
        return None
    c = dict(st[0])
    c["ambiguity"] = sum(1 for d in st if d["max_inl"] >= 25)
    c["any_hit"] = True
    return c


def classify(q, t_prov: int, t_conf: int, t_vote: int, n_vote: int,
             t_cells: int, max_amb: int):
    c = pick_candidate(q, t_vote)
    if c is None:
        return 0, None
    if c["max_inl"] >= t_conf and c["run"] >= n_vote and c["cells"] >= t_cells \
            and c["ambiguity"] <= max_amb:
        return 2, c
    if c["max_inl"] >= t_prov:
        return 1, c
    return 0, c


def _gt_index(G: dict):
    if G.get("_gt_by_label") is None:
        from research.tools.seqslam_probe import gt_segments_of
        labs = sorted({l for l in G["label"] if l and l != "None"})
        G["_gt_by_label"] = {L: gt_segments_of(L) for L in labs}
    return G["_gt_by_label"]


def is_positive(q, G: dict, min_sep: float) -> bool:
    """该查询 episode 是否存在更早的同地点访问（真值，**帧级**）。

    帧级口径：q 中某一帧 i 的 GT 标签 L 存在一个更早的 L 段，且其结束时间
    距 times[i] 至少 min_sep 秒。这样定义不依赖节点主导标签，避免把
    "节点横跨两个 GT 段" 误判成没有回环。
    """
    gt = _gt_index(G)
    for i in q["frames"]:
        L = G["label"][i]
        if not L or L == "None":
            continue
        t = float(G["times"][i])
        for a, b in gt.get(L, ()):
            if b + min_sep <= t:
                return True
    return False


def is_correct_frame(q, c, G: dict) -> bool:
    """帧级判定：产生最高内点的那一帧对，两侧 GT 标签相同。"""
    if c is None or c["best_j"] < 0 or c["best_i"] < 0:
        return False
    li = G["label"][c["best_i"]]
    lj = G["label"][c["best_j"]]
    if not li or li == "None" or not lj or lj == "None":
        return False
    return li == lj


def is_correct_node(q, c, G: dict) -> bool:
    """旧口径（有缺陷）：匹配节点的主导标签 == 查询节点的主导标签。

    当节点横跨两个 GT 段时会产生假阴性——实测 dancepool@59.1 正确匹配到
    真值同为 dancepool 的 t=36.65 帧，却因该节点主导标签是 pooldeck 被判"错误"。
    """
    return c is not None and G["ep_dom"][c["pi"]] == q["dom"]


def evaluate(G: dict, keys: list[int], t_prov, t_conf, t_vote, n_vote,
             t_cells, max_amb, min_sep: float) -> dict:
    """三级状态是**累积**的：confirmed ⊂ provisional ⊂ candidate。

    同时报告"仅到 provisional"与"累积到 provisional"两套数字——
    上一版只报了排他口径，导致 provisional_recall=0 而 confirmed_recall=0.44，
    表面自相矛盾。累积口径才是协议语义。
    """
    n_pos = 0
    conf_ok = conf_wrong = 0
    conf_ok_node = conf_wrong_node = 0
    ponly_ok = ponly_wrong = 0
    rej_no_cand = rej_below = 0
    per: dict[str, dict] = {}
    for qi in keys:
        q = G["per_query"][qi]
        if not is_positive(q, G, min_sep):
            continue
        n_pos += 1
        lvl, c = classify(q, t_prov, t_conf, t_vote, n_vote, t_cells, max_amb)
        L = q["dom"]
        d = per.setdefault(L, {"n": 0, "lvl2_ok": 0, "lvl2_wrong": 0,
                               "lvl2_ok_node": 0, "lvl2_wrong_node": 0,
                               "lvl1_ok": 0, "lvl1_wrong": 0,
                               "rej_no_cand": 0, "rej_below": 0,
                               "max_inl_p50": [], "run_p50": []})
        d["n"] += 1
        ok = is_correct_frame(q, c, G)
        okn = is_correct_node(q, c, G)
        if c is not None:
            d["max_inl_p50"].append(c["max_inl"])
            d["run_p50"].append(c["run"])
        if lvl == 2:
            if ok:
                conf_ok += 1
                d["lvl2_ok"] += 1
            else:
                conf_wrong += 1
                d["lvl2_wrong"] += 1
            if okn:
                conf_ok_node += 1
                d["lvl2_ok_node"] += 1
            else:
                conf_wrong_node += 1
                d["lvl2_wrong_node"] += 1
        elif lvl == 1:
            if ok:
                ponly_ok += 1
                d["lvl1_ok"] += 1
            else:
                ponly_wrong += 1
                d["lvl1_wrong"] += 1
        else:
            if c is None:
                rej_no_cand += 1
                d["rej_no_cand"] += 1
            else:
                rej_below += 1
                d["rej_below"] += 1

    for L, d in per.items():
        d["max_inl_p50"] = (round(float(np.median(d["max_inl_p50"])), 1)
                            if d["max_inl_p50"] else None)
        d["run_p50"] = (round(float(np.median(d["run_p50"])), 1)
                        if d["run_p50"] else None)
        d["prov_ok"] = d["lvl2_ok"] + d["lvl1_ok"]
        d["prov_wrong"] = d["lvl2_wrong"] + d["lvl1_wrong"]
    return {
        "n_pos_episodes": n_pos,
        "lvl2_conf_ok": conf_ok, "lvl2_conf_wrong": conf_wrong,
        "lvl1_prov_only_ok": ponly_ok, "lvl1_prov_only_wrong": ponly_wrong,
        "lvl0_rejected": rej_no_cand + rej_below,
        "lvl0_no_candidate": rej_no_cand, "lvl0_below_threshold": rej_below,
        # 累积口径（协议语义）
        "prov_ok_cum": conf_ok + ponly_ok, "prov_wrong_cum": conf_wrong + ponly_wrong,
        "provisional_recall": round((conf_ok + ponly_ok) / n_pos, 4) if n_pos else None,
        "provisional_wrong_rate": round((conf_wrong + ponly_wrong) / n_pos, 4) if n_pos else None,
        "confirmed_recall": round(conf_ok / n_pos, 4) if n_pos else None,
        "confirmed_wrong_rate": round(conf_wrong / n_pos, 4) if n_pos else None,
        "rejected_rate": round((rej_no_cand + rej_below) / n_pos, 4) if n_pos else None,
        "conf_ok": conf_ok, "conf_wrong": conf_wrong, "rejected": rej_no_cand + rej_below,
        # 旧口径（节点主导标签）——保留用于量化"归属假象"的幅度
        "conf_ok_node_rule": conf_ok_node, "conf_wrong_node_rule": conf_wrong_node,
        "confirmed_recall_node_rule": round(conf_ok_node / n_pos, 4) if n_pos else None,
        "confirmed_wrong_rate_node_rule": round(conf_wrong_node / n_pos, 4) if n_pos else None,
        # "全部拒绝"不能被当成成功
        "all_rejected": bool(conf_ok == 0 and ponly_ok == 0),
        "confirmed_zero_recall": bool(conf_ok == 0),
        "per_venue": per,
    }


def gt_segment_of_episodes(G: dict):
    """给每个 episode 标上它主要落在哪个 GT 访问段（仅用于诊断）。"""
    from research.tools.seqslam_probe import gt_segments_of
    labels = sorted({l for l in G["label"] if l and l != "None"})
    segs = []
    for lab in labels:
        for a, b in gt_segments_of(lab):
            segs.append((lab, float(a), float(b)))
    segs.sort(key=lambda x: x[1])
    out = []
    for qi in range(G["n_ep"]):
        fs = G["ep_frames"][qi]
        cnt: dict[int, int] = {}
        for i in fs:
            t = float(G["times"][i])
            for si, (lab, a, b) in enumerate(segs):
                if a - 1e-6 <= t <= b + 1e-6:
                    cnt[si] = cnt.get(si, 0) + 1
                    break
        out.append(max(cnt, key=cnt.get) if cnt else None)
    return out, segs


def same_visit_contamination(G: dict, min_sep: float) -> dict:
    """量化"节点边界污染"：有多少阳性 episode 的最近同地点证据其实来自**同一次访问**。

    如果切分把一个访问段切成两半、且间隔 > min_sep，后一半会被判为"阳性"，
    但它真正该连的是同一次访问的前一半——这正是节点边界错误污染协议证据的形式。
    """
    gt_of, segs = gt_segment_of_episodes(G)
    G["ep_gtseg"] = gt_of
    n_pos = 0
    contam = 0
    pure = 0
    detail = []
    for qi in range(G["n_ep"]):
        q = G["per_query"][qi]
        if not is_positive(q, G, min_sep):
            continue
        n_pos += 1
        same = [pi for pi in q["earlier"]
                if gt_of[pi] is not None and gt_of[pi] == gt_of[qi]]
        if same:
            contam += 1
            detail.append({"query_episode": qi, "t": [round(q["t0"], 2), round(q["t1"], 2)],
                           "label": q["dom"],
                           "same_visit_earlier_episodes": same,
                           "gt_segment": (segs[gt_of[qi]][1:3] if gt_of[qi] is not None else None)})
        else:
            pure += 1
    return {"n_positive_episodes": n_pos,
            "positives_with_same_visit_earlier_episode": contam,
            "positives_with_only_true_revisit_evidence": pure,
            "contamination_rate": round(contam / n_pos, 4) if n_pos else None,
            "detail": detail}


def load_episode_centroids(ep_path: Path, idx_path: Path, bow_path: Path):
    """每个 episode 的 BoW 质心（只用可用帧）。"""
    E = np.load(ep_path, allow_pickle=True)
    eps = E["episodes"].astype(np.int64)
    Z = np.load(idx_path, allow_pickle=True)
    row_of = Z["row_of"].astype(np.int64)
    BOW = np.load(bow_path).astype(np.float32)
    Bn = BOW / (np.linalg.norm(BOW, axis=1, keepdims=True) + 1e-12)
    cents, sizes = [], []
    for a, b in eps:
        fs = [int(i) for i in range(int(a), int(b)) if row_of[int(i)] >= 0]
        if not fs:
            cents.append(np.zeros(Bn.shape[1], np.float32))
            sizes.append(0)
            continue
        c = Bn[fs].mean(axis=0)
        cents.append((c / (np.linalg.norm(c) + 1e-12)).astype(np.float32))
        sizes.append(len(fs))
    return eps, cents, sizes


def merge_to_count(eps, cents, sizes, target: int):
    """自底向上把相邻 episode 合到只剩 target 个节点。

    每步合并**当前最相似的相邻对**（按 BoW 质心余弦），不使用真值。
    比"绝对相似度阈值"更稳：单链在绝对阈值下会把整段走线并成 1 个节点
    （实测 m<=0.8 时全部并成 1 个），说明 BoW 质心余弦的绝对水平不可用作粒度旋钮，
    只有相对排序可用。
    """
    eps = [[int(a), int(b)] for a, b in eps]
    cents = [c.copy() for c in cents]
    sizes = list(sizes)
    while len(eps) > target and len(eps) > 1:
        sims = [float(cents[i] @ cents[i + 1]) for i in range(len(eps) - 1)]
        i = int(np.argmax(sims))
        s = sizes[i] + sizes[i + 1]
        nc = cents[i] * sizes[i] + cents[i + 1] * sizes[i + 1]
        nc = nc / (np.linalg.norm(nc) + 1e-12) if s > 0 else cents[i]
        eps[i] = [eps[i][0], eps[i + 1][1]]
        cents[i] = nc.astype(np.float32)
        sizes[i] = s
        del eps[i + 1], cents[i + 1], sizes[i + 1]
    return np.asarray(eps, dtype=np.int64), np.asarray(sizes)


def _uf(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return [find(i) for i in range(n)]


def candidate_margin(st) -> float:
    """最优 / 次优 内点比。无第二假设视为无穷大（决定性）。"""
    if len(st) < 2 or st[0]["max_inl"] <= 0:
        return float("inf")
    return st[0]["max_inl"] / max(1, st[1]["max_inl"])


def visual_seed_edges(G: dict, keys: list[int], t_prov, t_conf, t_vote, n_vote,
                      min_sep: float, *, seed_margin: float = 2.0,
                      seed_level: str = "visual_decisive") -> dict[int, int]:
    """复算共识的**视觉播种边**（与 :func:`consensus_assign` 内部规则逐条一致）。

    存在的理由：`route_history_seed_edges()` 需要"高置信视觉边"作为输入才能叠加
    路线历史；把它单独抽出来可以让二者共用同一份规则，避免闸门与共识各算一套、
    悄悄漂移。**它不改变共识内部逻辑**——共识在没有外部 seed_edges 时仍自行播种。
    """
    out: dict[int, int] = {}
    for qi in keys:
        q = G["per_query"][qi]
        if not is_positive(q, G, min_sep):
            continue
        lvl, c = classify(q, t_prov, t_conf, t_vote, n_vote, 0, 99)
        if c is None:
            continue
        if seed_level == "provisional":
            ok = lvl >= 1
        elif seed_level == "all_confirmed":
            ok = lvl == 2
        else:                                   # visual_decisive
            ok = (lvl == 2
                  and candidate_margin(cand_stats(q, t_vote)) >= seed_margin)
        if ok:
            out[qi] = c["pi"]
    return out


def load_route_history(action_log: Path, episodes) -> list[dict]:
    """离线读取动作日志并按 episode 汇总成路线历史。

    `driver_log.py` 位于包 `neko_anyadance_body` 根，使用相对导入，必须按包名导入。
    这里按 `tests/_bootstrap.py` 同样的方式注册一个合成包模块，不依赖安装。
    """
    import types

    pkg = Path(__file__).resolve().parents[2]
    if "neko_anyadance_body" not in sys.modules:
        mod = types.ModuleType("neko_anyadance_body")
        mod.__path__ = [str(pkg)]              # type: ignore[attr-defined]
        sys.modules["neko_anyadance_body"] = mod
    from neko_anyadance_body.driver_log import (  # noqa: PLC0415
        episode_action_summary, load_action_timeline)

    tb, rows = load_action_timeline(action_log)
    if tb is None and not rows:
        # 文件不存在 / 全是坏行：当作"没有日志"，让闸门走显式降级，而不是报错。
        return []
    eps = [[int(a), int(b)] for a, b in np.asarray(episodes).tolist()]
    fps = tb.fps if tb is not None else 20.0
    return episode_action_summary(action_log, eps, fps=fps)


def fold_independent_evidence(members, ep_t, contrib_cap: int,
                              indep_gap_s: float):
    """把一个簇内多个候选 episode 折成"独立证据之和"（用户 2026-09-21 定稿）。

    三条限制：
    - 同一 episode 只贡献一次（`members` 本身已按 episode 去重）；
    - 单条贡献封顶 ``contrib_cap``（<=0 表示不封顶）——否则一个大簇会天然压过小簇，
      即使每条证据都很弱；
    - 同一访问时段内的相邻候选不得重复累加：按贡献降序贪心，只接受与已接受者
      时间间隔 >= ``indep_gap_s`` 的候选。

    返回 ``(score, n_independent, member_pi_list)``。
    """
    scored = []
    for d in members:
        v = d["max_inl"] if contrib_cap <= 0 else min(d["max_inl"], contrib_cap)
        scored.append((v, d))
    scored.sort(key=lambda x: (-x[0], x[1]["pi"]))
    accepted: list[tuple[int, dict]] = []
    for v, d in scored:
        t0, t1 = ep_t[d["pi"]]
        if all(
            (t1 + indep_gap_s <= a0) or (a1 + indep_gap_s <= t0)
            for _, a in accepted
            for a0, a1 in [ep_t[a["pi"]]]
        ):
            accepted.append((v, d))
    return (sum(v for v, _ in accepted), len(accepted),
            [a["pi"] for _, a in accepted])


def consensus_assign(G: dict, keys: list[int], t_prov, t_conf, t_vote, n_vote,
                     min_sep: float, iters: int = 4, contrib_cap: int = 100,
                     indep_gap_s: float = 8.0, seed_margin: float = 2.0,
                     seed_level: str = "visual_decisive",
                     inject: dict[int, int] | None = None,
                     seed_edges: dict[int, int] | None = None,
                     seed_source_hint: str | None = None,
                     route_history_available: bool | None = None):
    # 注意：seed_source_hint / route_history_available 用于把"种子来自哪条通道"
    # 的显式溯源透传进来（路线历史闸门降级为视觉代理时也要如实标注）。二者都
    # 是**可加、向后兼容**的可选参数：不传时行为与旧版完全一致。
    """图结构共识：候选证据**按已确认簇折叠求和**，而不是取单个最大内点。

    动机（ep39 反例）：`ep39` 的假设是
        ep23(pooldeck 帧) 205 / ep30(hall 帧) 126 / ep31(hall 帧) 89 / ep00(hall 帧) 84
    argmax 取 205 → 合并到 pooldeck（错）。但 ep30/ep31/ep00 之间**已经被确认边连成一个簇**
    （它们是同一个地点），把它们的内点**折叠求和**得 299 > 205 → 归到正确的簇。

    只用已确认边与内点数，不使用真值标签，也不使用任何朝向积分。

    三道生产保护（用户 2026-09-21 定稿）
    ------------------------------------
    1. **封顶 + 独立证据**（``contrib_cap`` / ``indep_gap_s``）：见
       :func:`fold_independent_evidence`。防止"大簇天然压过小簇"。
    2. **只有高置信边才能播种/合并簇**（``seed_level`` / ``seed_margin``）：
       - ``visual_decisive``（默认）：``confirmed_visual_loop`` 且最优/次优内点比 >=
         ``seed_margin``。**歧义的 confirmed_visual_loop（如 ep39，比值 0.61）不得播种**，
         否则错误边会自举成错误地点簇。
       - ``all_confirmed``：只要是 confirmed 就播种（不推荐，用于对照）。
       - ``provisional``：连 provisional 也播种（对照用，证明保护确实有效）。
       外部通道（路线历史 / 语义地标）可通过 ``seed_edges`` 直接给出
       ``confirmed_place_identity`` 级别的播种边——这是唯一被允许的"真·播种"。
    3. **可撤销**：本函数只产出 ``identity_hypothesis``；是否提升为稳定地点身份由
       :func:`promotion_report` 按"独立证据 >= 2 个访问时段"决定。新证据冲突时保留
       alias，不覆盖。

    ``inject`` 用于人工错误播种回归测试：强制某些查询指向指定 episode。
    """
    pos = [q for q in keys if is_positive(G["per_query"][q], G, min_sep)]
    ep_t = G["ep_t"]

    def seed_ok(qi: int, lvl: int, c) -> bool:
        if c is None:
            return False
        if seed_level == "provisional":
            return lvl >= 1
        if seed_level == "all_confirmed":
            return lvl == 2
        if lvl != 2:                      # visual_decisive
            return False
        st = cand_stats(G["per_query"][qi], t_vote)
        return candidate_margin(st) >= seed_margin

    assign: dict[int, int] = {}
    seed_source_meta = "visual_proxy"
    seed_list: list[int] = []
    if seed_edges:
        assign.update({int(k): int(v) for k, v in seed_edges.items()})
        # 显式溯源：调用方已声明种子来自哪条通道（含降级为视觉代理）时如实标注，
        # 不再一律臆造 "external(route/semantic)"——否则降级路径会被误标为可用通道。
        seed_source_meta = (seed_source_hint if seed_source_hint is not None
                            else "external(route/semantic)")
    else:
        for qi in pos:
            lvl, c = classify(G["per_query"][qi], t_prov, t_conf, t_vote,
                              n_vote, 0, 99)
            if seed_ok(qi, lvl, c):
                assign[qi] = c["pi"]
                seed_list.append(qi)
    injected = {int(k): int(v) for k, v in (inject or {}).items()}
    assign.update(injected)

    hist: list[dict] = []
    detail: dict[int, dict] = {}
    for _ in range(iters):
        cl = _uf(G["n_ep"], list(assign.items()))
        new: dict[int, int] = {}
        for qi in pos:
            if qi in injected:
                new[qi] = injected[qi]
                continue
            st = cand_stats(G["per_query"][qi], t_vote)
            if not st:
                continue
            by_r: dict[int, list[dict]] = {}
            for d in st:
                by_r.setdefault(cl[d["pi"]], []).append(d)
            scores = {r: fold_independent_evidence(ms, ep_t, contrib_cap,
                                                   indep_gap_s)
                      for r, ms in by_r.items()}
            best_r = max(scores, key=lambda r: (scores[r][0], scores[r][1], -r))
            sc, nind, members = scores[best_r]
            mset = set(members)
            pick = max([d for d in by_r[best_r] if d["pi"] in mset],
                       key=lambda d: d["max_inl"])
            runner = sorted((s for r, s in scores.items() if r != best_r),
                            reverse=True)
            detail[qi] = {
                "cluster": best_r, "cluster_score": sc,
                "cluster_score_runner_up": (runner[0][0] if runner else 0),
                "n_independent_visits": nind,
                "independent_members": members,
                "pick": pick["pi"], "pick_inl": pick["max_inl"],
            }
            if pick["max_inl"] >= t_conf:
                new[qi] = pick["pi"]
        hist.append({"n_assigned": len(new),
                     "changed": sum(1 for k in new
                                    if assign.get(k) != new[k])})
        if new == assign:
            break
        assign = new
    detail["_meta"] = {"contrib_cap": contrib_cap, "indep_gap_s": indep_gap_s,
                       "seed_margin": seed_margin, "seed_level": seed_level,
                       "seed_source": seed_source_meta,
                       "route_history_available": (
                           route_history_available
                           if route_history_available is not None
                           else (seed_source_meta == "visual+route_history")),
                       "n_injected": len(injected),
                       "seed_episodes": sorted(set(seed_list))}
    return assign, hist, detail


def promotion_report(G: dict, assign: dict[int, int], detail: dict,
                     min_independent: int = 2) -> dict:
    """把共识结果标成可撤销的 ``identity_hypothesis``，并决定是否提升。

    提升条件（本素材可用的口径）：**独立证据 >= ``min_independent`` 个访问时段**。
    未达标的保留为 alias（多假设节点），不合并 PlaceIdentity。
    这是"可撤销"的落点：冲突证据出现时只要删掉对应 seed_edge 即可回到 hypothesis。
    """
    promoted, hypotheses = [], []
    for qi, pi in sorted(assign.items()):
        if qi not in detail or qi == "_meta":
            continue
        d = detail[qi]
        row = {"query_episode": qi, "assigned": pi,
               "cluster": d["cluster"], "cluster_score": d["cluster_score"],
               "cluster_score_runner_up": d["cluster_score_runner_up"],
               "n_independent_visits": d["n_independent_visits"],
               "independent_members": d["independent_members"],
               "state": "identity_hypothesis",
               "may_merge_place_identity": d["n_independent_visits"] >= min_independent,
               "promotion_reason": ("多访问时段独立证据 >= %d" % min_independent
                                    if d["n_independent_visits"] >= min_independent
                                    else "独立证据不足，保留 alias")}
        (promoted if row["may_merge_place_identity"] else hypotheses).append(row)
    return {"rule": "独立访问时段数 >= %d 才提升为 confirmed_place_identity 候选" % min_independent,
            "meta": detail.get("_meta", {}),
            "n_hypothesis": len(promoted) + len(hypotheses),
            "n_promoted": len(promoted), "n_alias_retained": len(hypotheses),
            "promoted": promoted, "alias_retained": hypotheses}


def evaluate_assign(G: dict, keys: list[int], assign: dict, min_sep: float,
                    t_vote: int) -> dict:
    """按给定指派评定：正确 / 错误 / 未指派。"""
    n_pos = ok = wrong = unassigned = 0
    rows = []
    for qi in keys:
        q = G["per_query"][qi]
        if not is_positive(q, G, min_sep):
            continue
        n_pos += 1
        pi = assign.get(qi)
        if pi is None:
            unassigned += 1
            continue
        st = cand_stats(q, t_vote)
        hit = [d for d in st if d["pi"] == pi]
        if not hit:
            unassigned += 1
            continue
        c = hit[0]
        good = is_correct_frame(q, c, G)
        ok += good
        wrong += (not good)
        rows.append({"query_episode": qi, "assigned": pi,
                     "t": [round(q["t0"], 2), round(q["t1"], 2)],
                     "max_inl": c["max_inl"],
                     "assigned_gt": G["label"][c["best_j"]] if c["best_j"] >= 0 else None,
                     "verdict": "OK" if good else "WRONG"})
    return {"n_pos": n_pos, "ok": ok, "wrong": wrong, "unassigned": unassigned,
            "recall": round(ok / n_pos, 4) if n_pos else None,
            "wrong_rate": round(wrong / n_pos, 4) if n_pos else None,
            "rows": rows}


# --------------------------------------------------------- 路线历史 → 播种边
def route_history_seed_edges(
    visual_seeds: dict[int, int],
    summaries: list[dict] | None,
    *,
    min_records: int = 2,
    min_path_m: float = 0.05,
) -> dict:
    """把路线历史作为**必要条件**叠加到高置信视觉种子上。

    设计依据（已定稿）："seed_edges 只使用高置信视觉边加路线历史；错误视觉边不能
    单独播种地点簇"。所以本函数**只做过滤，永不新增**——不会凭空造出视觉上没有的边。

    ``summaries`` 来自 ``driver_log.episode_action_summary()``。为 None / 空 表示
    **本次录制没有动作日志**：此时按既定规则**自动降级为纯视觉代理**，而不是伪造
    一份"路线历史支持"——降级是显式的（``source`` 说明用了哪条通道）。

    保留的两条一致性都是**有据可依的必要条件**：
    1. 两端 episode 都真的观察到动作（``n_records >= min_records``）；
    2. 两端累计前进距离都 ``> min_path_m``（确实走过路，不是原地站着）。

    ⚠️ 这是必要条件，**不是**充分条件：它挡得住"根本没走过"的边，挡不住
    ``ep39`` 那种"两端都走过、只是视觉不可分辨"的边。真正的顺序/距离一致性必须
    等一份**新录制**（视频 + 帧索引 + 动作日志同时落盘）才能标定。
    """
    seeds = {int(k): int(v) for k, v in (visual_seeds or {}).items()}
    if not summaries:
        return {
            "available": False, "source": "visual_proxy",
            "reason": "no_action_log", "seed_edges": dict(seeds),
            "n_visual": len(seeds), "n_kept": len(seeds), "dropped": [],
            "note": "无动作日志：按规则降级为视觉代理，不伪造路线历史",
        }
    by_ep = {int(s.get("episode", -1)): s for s in summaries}
    kept: dict[int, int] = {}
    dropped: list[dict] = []
    for qi, pi in seeds.items():
        a, b = by_ep.get(qi), by_ep.get(pi)
        reason = None
        if a is None or b is None:
            reason = "episode_missing_from_log"
        elif (int(a.get("n_records", 0)) < min_records
              or int(b.get("n_records", 0)) < min_records):
            reason = "insufficient_action_records"
        elif (float(a.get("osc_forward_distance", 0.0)) <= min_path_m
              or float(b.get("osc_forward_distance", 0.0)) <= min_path_m):
            reason = "no_observed_path"
        if reason is None:
            kept[qi] = pi
        else:
            dropped.append({"query": qi, "candidate": pi, "reason": reason})
    return {
        "available": True, "source": "visual+route_history",
        "seed_edges": kept, "dropped": dropped,
        "n_visual": len(seeds), "n_kept": len(kept),
    }


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=Path, default=Path(".tmp/episodes.npz"))
    ap.add_argument("--index", type=Path, default=Path(".tmp/loop_frame_index.npz"))
    ap.add_argument("--min-sep", type=float, default=8.0)
    ap.add_argument("--t-prov", type=int, default=25)
    ap.add_argument("--t-conf", type=int, default=40)
    ap.add_argument("--t-vote", type=int, default=25)
    ap.add_argument("--n-vote", type=int, default=2)
    ap.add_argument("--t-cells", type=int, default=0,
                    help="confirmed 附加条件（0=不启用；§17 已证 cells>=6 过严）")
    ap.add_argument("--max-amb", type=int, default=99,
                    help="允许的竞争地点数上限（重复纹理风险代理）")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--merge-sweep", action="store_true")
    ap.add_argument("--dump-decisions", action="store_true")
    ap.add_argument("--aliases", action="store_true",
                    help="输出多假设（别名）节点：每个接受边的全部竞争地点")
    ap.add_argument("--amb-ratio", type=float, default=0.5,
                    help="次优/最优 内点比 >= 该值即标记 visual_ambiguity=high")
    ap.add_argument("--consensus", action="store_true",
                    help="图结构共识：候选证据按已确认簇折叠求和后再指派")
    ap.add_argument("--contrib-cap", type=int, default=100,
                    help="簇内单个 episode 贡献上限（<=0 = 不封顶）")
    ap.add_argument("--indep-gap", type=float, default=8.0,
                    help="簇内两条候选视为'独立访问'所需的最小时间间隔（秒）")
    ap.add_argument("--seed-margin", type=float, default=2.0,
                    help="播种边要求的最优/次优内点比（visual_decisive 模式）")
    ap.add_argument("--seed-level", default="visual_decisive",
                    choices=("visual_decisive", "all_confirmed", "provisional"),
                    help="簇播种所需的最低证据级别")
    ap.add_argument("--inject-wrong-seed", default=None,
                    help="人工错误播种，格式 'qep:pi,qep:pi'（错误播种回归测试用）")
    ap.add_argument("--action-log", type=Path, default=None,
                    help="动作时间轴 JSONL；给出后视觉种子会再经路线历史闸门过滤"
                         "（无文件/空日志 ⇒ 显式降级为视觉代理，不伪造）")
    ap.add_argument("--promotion-min-independent", type=int, default=2,
                    help="提升为稳定地点身份所需的独立访问时段数")
    ap.add_argument("--bow", type=Path,
                    default=Path(".slam_probe/offline_probe/bow_full_v400.npy"))
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    G = build_episode_evidence(args.episodes, args.index, args.min_sep)
    n_ep = G["n_ep"]
    all_keys = list(range(n_ep))

    pos_eps = [i for i in all_keys if is_positive(G["per_query"][i], G, args.min_sep)]
    dom_cnt: dict[str, int] = {}
    for d in G["ep_dom"]:
        if d:
            dom_cnt[d] = dom_cnt.get(d, 0) + 1

    res: dict = {
        "protocol": "candidate -> provisional_loop -> confirmed_loop（episode 级、偏移无关）",
        "place_unit": "PlaceEpisode（episode_segment.py 在线切分，无真值）",
        "vote_form": "place-level over episodes, offset-agnostic",
        "episodes": {"n": n_ep, "dominant_labels": dom_cnt,
                     "n_positive_episodes": len(pos_eps),
                     "note": "阳性 = 存在更早的同一 dominant label 的 episode"},
        "reference_config": {"t_prov": args.t_prov, "t_conf": args.t_conf,
                             "t_vote": args.t_vote, "n_vote": args.n_vote,
                             "t_cells": args.t_cells, "max_amb": args.max_amb},
    }

    ref = evaluate(G, all_keys, args.t_prov, args.t_conf, args.t_vote,
                   args.n_vote, args.t_cells, args.max_amb, args.min_sep)
    res["reference"] = ref
    res["node_boundary_contamination"] = same_visit_contamination(G, args.min_sep)

    # 逐地点明细（含该地点 episode 数）
    pv = {}
    for L, d in ref["per_venue"].items():
        pv[L] = {**d, "n_episodes_of_venue": dom_cnt.get(L, 0)}
    res["per_venue_reference"] = pv

    if args.grid:
        grid = []
        for t_conf in (20, 30, 40, 60, 80, 120):
            for n_vote in range(0, 4):
                e = evaluate(G, all_keys, args.t_prov, t_conf, args.t_vote,
                             n_vote, args.t_cells, args.max_amb, args.min_sep)
                grid.append({"t_conf": t_conf, "n_vote": n_vote,
                             "prov_recall": e["provisional_recall"],
                             "prov_wrong": e["provisional_wrong_rate"],
                             "conf_recall": e["confirmed_recall"],
                             "conf_wrong": e["confirmed_wrong_rate"],
                             "conf_ok": e["conf_ok"], "conf_wrong_n": e["conf_wrong"],
                             "rejected": e["rejected"]})
        res["grid"] = grid

        # n_vote 边际效果（逐地点，t_conf=20）
        marg = {}
        for n_vote in range(0, 4):
            e = evaluate(G, all_keys, args.t_prov, 20, args.t_vote,
                         n_vote, args.t_cells, args.max_amb, args.min_sep)
            marg[f"n_vote={n_vote}"] = {
                L: {"recall": round(d["lvl2_ok"] / d["n"], 4) if d["n"] else None,
                    "wrong": round(d["lvl2_wrong"] / d["n"], 4) if d["n"] else None}
                for L, d in e["per_venue"].items()}
        res["n_vote_effect_per_venue_at_t_conf20"] = marg

        # leave-one-loop-out：留出地点上重新选参（训练集需 conf_wrong == 0）
        venues = sorted({d for d in G["ep_dom"] if d})
        lolo = {}
        for held in venues:
            tr = [i for i in all_keys if G["ep_dom"][i] != held]
            te = [i for i in all_keys if G["ep_dom"][i] == held]
            te_pos = [i for i in te if is_positive(G["per_query"][i], G, args.min_sep)]
            if not te_pos:
                lolo[held] = {"n": 0, "note": "该地点无阳性 episode，无法做留出"}
                continue
            best = None
            n_zero_wrong = 0
            n_usable = 0
            for t_conf in (20, 30, 40, 60, 80, 120):
                for n_vote in range(0, 4):
                    e = evaluate(G, tr, args.t_prov, t_conf, args.t_vote,
                                 n_vote, args.t_cells, args.max_amb, args.min_sep)
                    if e["conf_wrong"] > 0 or e["confirmed_recall"] is None:
                        continue
                    n_zero_wrong += 1
                    if e["conf_ok"] > 0:
                        n_usable += 1
                    key = (e["conf_ok"], -n_vote, -t_conf)
                    if best is None or key > best[0]:
                        best = (key, (t_conf, n_vote), e)
            if best is None:
                lolo[held] = {"n": len(te_pos),
                              "note": "训练集上不存在 conf_wrong=0 的配置"}
                continue
            _, (tc, nv), etr = best
            ete = evaluate(G, te_pos, args.t_prov, tc, args.t_vote, nv,
                           args.t_cells, args.max_amb, args.min_sep)
            lolo[held] = {
                "n": len(te_pos), "chosen": {"t_conf": tc, "n_vote": nv},
                "train_zero_wrong_configs": n_zero_wrong,
                "train_zero_wrong_configs_with_recall": n_usable,
                "train": {"conf_recall": etr["confirmed_recall"],
                          "conf_wrong": etr["confirmed_wrong_rate"]},
                "held_out": {"conf_recall": ete["confirmed_recall"],
                             "conf_wrong": ete["confirmed_wrong_rate"],
                             "conf_ok": ete["conf_ok"], "conf_wrong_n": ete["conf_wrong"],
                             "rejected": ete["rejected"],
                             "prov_recall": ete["provisional_recall"]},
                "selector_degenerate": bool(n_usable == 0),
            }
        res["leave_one_loop_out"] = lolo

    if args.aliases:
        nodes = []
        for qi in all_keys:
            q = G["per_query"][qi]
            if not is_positive(q, G, args.min_sep):
                continue
            lvl, c = classify(q, args.t_prov, args.t_conf, args.t_vote,
                              args.n_vote, args.t_cells, args.max_amb)
            if lvl < 2 or c is None:
                continue
            st = cand_stats(q, args.t_vote)[:4]
            ratio = (st[1]["max_inl"] / st[0]["max_inl"]) if len(st) > 1 else 0.0
            hyps = []
            for d in st:
                hyps.append({
                    "candidate_episode": d["pi"],
                    "candidate_window": [round(G["ep_t"][d["pi"]][0], 2),
                                         round(G["ep_t"][d["pi"]][1], 2)],
                    "max_inliers": d["max_inl"], "support_run": d["run"],
                    "cells": d["cells"], "remaining": d["rem"],
                    "gt_at_best_frame": (G["label"][d["best_j"]]
                                         if d["best_j"] >= 0 else None),
                })
            nodes.append({
                "query_episode": qi,
                "query_window": [round(q["t0"], 2), round(q["t1"], 2)],
                "state": "confirmed_visual_loop",
                "may_merge_place_identity": False,
                "visual_ambiguity": "high" if ratio >= args.amb_ratio else "low",
                "runner_up_ratio": round(ratio, 3),
                "possible_places": hyps,
                "note": "独立证据（路线历史/语义地标）未实现前不得合并 PlaceIdentity",
            })
        res["alias_nodes"] = nodes
        print("\n--- alias nodes（多假设，不合并）---")
        for n in nodes:
            print("  episode_%02d  %s   ambiguity=%s (ratio=%.2f)" % (
                n["query_episode"],
                "%.2f-%.2f" % tuple(n["query_window"]),
                n["visual_ambiguity"], n["runner_up_ratio"]))
            for h in n["possible_places"]:
                print("      ├─ possible_place: ep%02d %s  inl=%-4d run=%-3d "
                      "GT@best=%-9s remaining=%d" % (
                          h["candidate_episode"],
                          "%.2f-%.2f" % tuple(h["candidate_window"]),
                          h["max_inliers"], h["support_run"],
                          h["gt_at_best_frame"], h["remaining"]))
            if len(n["possible_places"]) < 2:
                print("      └─ (无第二假设)")

    if args.consensus:
        inject = None
        if args.inject_wrong_seed:
            inject = {}
            for chunk in args.inject_wrong_seed.split(","):
                if not chunk.strip():
                    continue
                qi_s, pi_s = chunk.split(":")
                inject[int(qi_s)] = int(pi_s)
        # 路线历史闸门（只过滤、不新增）：给出 --action-log 时，把视觉高置信种子
        # 再经 episode_action_summary 的路线历史过滤一遍。没有日志时闸门**显式降级**
        # 为视觉代理并把原边原样返回——绝不能因此凭空造出视觉上不存在的边。
        route = None
        if args.action_log is not None:
            summaries = load_route_history(args.action_log, G["eps"])
            vis = visual_seed_edges(
                G, all_keys, args.t_prov, args.t_conf, args.t_vote,
                args.n_vote, args.min_sep, seed_margin=args.seed_margin,
                seed_level=args.seed_level)
            route = route_history_seed_edges(vis, summaries)
        asg, hist, cdet = consensus_assign(
            G, all_keys, args.t_prov, args.t_conf, args.t_vote, args.n_vote,
            args.min_sep, contrib_cap=args.contrib_cap,
            indep_gap_s=args.indep_gap, seed_margin=args.seed_margin,
            seed_level=args.seed_level, inject=inject,
            seed_edges=(route["seed_edges"] if route is not None else None),
            seed_source=(route["source"] if route is not None else None),
            route_history_available=(route["available"] if route is not None
                                     else None))
        cons = evaluate_assign(G, all_keys, asg, args.min_sep, args.t_vote)
        # 对照：纯 argmax 指派
        base = {}
        for qi in all_keys:
            q = G["per_query"][qi]
            if not is_positive(q, G, args.min_sep):
                continue
            st = cand_stats(q, args.t_vote)
            if st and st[0]["max_inl"] >= args.t_conf:
                base[qi] = st[0]["pi"]
        basel = evaluate_assign(G, all_keys, base, args.min_sep, args.t_vote)
        promo = promotion_report(G, asg, cdet,
                                 args.promotion_min_independent)
        res["consensus"] = {
            "rule": "候选内点按已确认簇折叠求和（封顶 + 独立访问 + 高置信播种），迭代至稳定",
            "guards": cdet.get("_meta", {}),
            "route_history": route,
            "iterations": hist,
            "promotion": {k: promo[k] for k in
                          ("rule", "n_hypothesis", "n_promoted", "n_alias_retained")},
            "argmax": {"recall": basel["recall"], "wrong_rate": basel["wrong_rate"],
                       "ok": basel["ok"], "wrong": basel["wrong"],
                       "unassigned": basel["unassigned"]},
            "consensus": {"recall": cons["recall"], "wrong_rate": cons["wrong_rate"],
                          "ok": cons["ok"], "wrong": cons["wrong"],
                          "unassigned": cons["unassigned"]},
            "rows": cons["rows"],
            "changed_rows": [r for r in cons["rows"]
                             if base.get(r["query_episode"]) != r["assigned"]],
            "promotion_detail": promo,
        }
        print("\n--- 图结构共识 vs argmax ---")
        print("  argmax   : ok=%d wrong=%d unassigned=%d  recall=%.3f wrong=%.3f" % (
            basel["ok"], basel["wrong"], basel["unassigned"],
            basel["recall"] or 0, basel["wrong_rate"] or 0))
        print("  consensus: ok=%d wrong=%d unassigned=%d  recall=%.3f wrong=%.3f" % (
            cons["ok"], cons["wrong"], cons["unassigned"],
            cons["recall"] or 0, cons["wrong_rate"] or 0))
        print("  迭代:", hist)
        print("  保护: %s" % cdet.get("_meta", {}))
        if route is not None:
            print("  路线历史: source=%s available=%s 视觉=%d 保留=%d 丢弃=%d" % (
                route.get("source"), route.get("available"),
                route.get("n_visual", 0), route.get("n_kept", 0),
                len(route.get("dropped", []))))
            for d in route.get("dropped", []):
                print("      - qep%d -> ep%d 丢弃: %s" % (
                    d["query"], d["candidate"], d["reason"]))
        elif args.action_log is not None:
            print("  路线历史: 未启用（未给出 --action-log ⇒ 视觉代理）")
        print("  %-4s %-14s %-10s %-12s %-6s %-6s %s" % (
            "qep", "window", "argmax→", "consensus→", "inl", "GT@best", "verdict"))
        for r in cons["rows"]:
            print("  %-4d %6.2f-%6.2f %-10s %-12s %-6d %-6s %s" % (
                r["query_episode"], r["t"][0], r["t"][1],
                str(base.get(r["query_episode"])), str(r["assigned"]),
                r["max_inl"], r["assigned_gt"], r["verdict"]))
        print("\n--- identity_hypothesis / 提升（可撤销）---")
        print("  %-4s %-10s %-7s %-7s %-6s %s" % (
            "qep", "assigned→", "score", "runner", "n_indep", "state"))
        for row in promo["promoted"] + promo["alias_retained"]:
            print("  %-4d %-10s %-7d %-7d %-6d %s" % (
                row["query_episode"], str(row["assigned"]), row["cluster_score"],
                row["cluster_score_runner_up"], row["n_independent_visits"],
                "confirmed_place_identity(candidate)"
                if row["may_merge_place_identity"] else "alias_retained"))
        print("  提升 %d / 保留 alias %d（%s）" % (
            promo["n_promoted"], promo["n_alias_retained"], promo["rule"]))

    if args.merge_sweep:
        eps0, cents, sizes = load_episode_centroids(args.episodes, args.index, args.bow)
        lbl = G["label"]
        rows_ms = []
        for n_target in (40, 36, 32, 28, 24, 20, 16, 12, 8):
            meps, msz = merge_to_count(eps0, cents, sizes, n_target)
            G2 = build_episode_evidence(args.episodes, args.index, args.min_sep,
                                        eps_override=meps)
            keys2 = list(range(len(meps)))
            n_pos2 = sum(1 for i in keys2
                         if is_positive(G2["per_query"][i], G2, args.min_sep))
            # 纯度（只用 label，不用 GT 段边界）
            pur = []
            for a, b in meps:
                cnt: dict[str, int] = {}
                for i in range(int(a), int(b)):
                    L = lbl[i]
                    if L and L != "None":
                        cnt[L] = cnt.get(L, 0) + 1
                if cnt:
                    pur.append(max(cnt.values()) / sum(cnt.values()))
            e = evaluate(G2, keys2, args.t_prov, args.t_conf, args.t_vote,
                         args.n_vote, args.t_cells, args.max_amb, args.min_sep)
            rows_ms.append({
                "target_nodes": n_target, "n_episodes": int(len(meps)),
                "n_pos": n_pos2,
                "purity": round(float(np.mean(pur)), 4) if pur else None,
                "conf_ok": e["conf_ok"], "conf_wrong": e["conf_wrong"],
                "confirmed_recall": e["confirmed_recall"],
                "confirmed_wrong_rate": e["confirmed_wrong_rate"],
                "prov_recall": e["provisional_recall"],
                "prov_wrong": e["provisional_wrong_rate"],
                "rejected": e["rejected"]})
            print(f"[merge] target={n_target:>3} -> eps={len(meps):>3} pos={n_pos2:>3} "
                  f"purity={rows_ms[-1]['purity']} "
                  f"conf={e['conf_ok']}/{e['conf_wrong']} "
                  f"({e['confirmed_recall']}/{e['confirmed_wrong_rate']}) "
                  f"prov={e['provisional_recall']}/{e['provisional_wrong_rate']}")
        res["merge_sweep"] = {"gt_segments": 16, "rows": rows_ms,
                              "note": "按相邻对 BoW 质心余弦自底向上合并到目标节点数；"
                                      "只用相对排序，不使用真值"}

    if args.dump_decisions:
        rows = []
        for qi in all_keys:
            q = G["per_query"][qi]
            if not is_positive(q, G, args.min_sep):
                continue
            is_same_visit = (G.get("ep_gtseg") is not None
                             and G["ep_gtseg"][qi] is not None)
            lvl, c = classify(q, args.t_prov, args.t_conf, args.t_vote,
                              args.n_vote, args.t_cells, args.max_amb)
            rows.append({
                "query_ep": qi, "t": [round(q["t0"], 2), round(q["t1"], 2)],
                "label": q["dom"], "level": lvl,
                "matched_ep": (c["pi"] if c else None),
                "matched_label": (G["ep_dom"][c["pi"]] if c else None),
                "matched_t": ([round(G["ep_t"][c["pi"]][0], 2),
                               round(G["ep_t"][c["pi"]][1], 2)] if c else None),
                "max_inl": (c["max_inl"] if c else None),
                "run": (c["run"] if c else None),
                "cells": (c["cells"] if c else None),
                "ambiguity": (c["ambiguity"] if c else None),
                "best_i": (c["best_i"] if c else None),
                "best_j": (c["best_j"] if c else None),
                "best_i_gt": (G["label"][c["best_i"]] if c and c["best_i"] >= 0 else None),
                "best_j_gt": (G["label"][c["best_j"]] if c and c["best_j"] >= 0 else None),
                "best_j_t": (round(float(G["times"][c["best_j"]]), 2)
                             if c and c["best_j"] >= 0 else None),
                "verdict": ("OK" if (c and lvl == 2 and is_correct_frame(q, c, G)) else
                            ("WRONG" if (c and lvl == 2) else
                             ("prov_ok" if (lvl == 1 and is_correct_frame(q, c, G))
                              else ("prov_wrong" if lvl == 1 else "reject")))),
                "verdict_node_rule": ("OK" if (c and lvl == 2 and is_correct_node(q, c, G)) else
                                      ("WRONG" if (c and lvl == 2) else
                                       ("prov_ok" if (lvl == 1 and is_correct_node(q, c, G))
                                        else ("prov_wrong" if lvl == 1 else "reject")))),
            })
        res["decisions"] = rows
        print("\n--- decisions (positive episodes) ---")
        print("  qep  window(s)        node_lab   -> matched         inl  run  amb  "
              "best_pair(GT)                  verdict   node-rule")
        for r in rows:
            mt = ("ep%02d@%.1f-%.1f" % (r["matched_ep"], r["matched_t"][0],
                                        r["matched_t"][1])) if r["matched_ep"] is not None else "-"
            bp = ("f%d(%s) <-> f%d(%s)@%.1f" % (
                r["best_i"] or -1, r["best_i_gt"], r["best_j"] or -1, r["best_j_gt"],
                r["best_j_t"] or -1)) if r["best_j"] is not None else "-"
            print("  %3d  %5.2f-%5.2f  %-10s -> %-14s %-4s %-4s %-4s %-30s %-9s %s" % (
                r["query_ep"], r["t"][0], r["t"][1], r["label"], mt,
                r["max_inl"], r["run"], r["ambiguity"], bp,
                r["verdict"], r["verdict_node_rule"]))

    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_venue_reference", "grid",
                                   "n_vote_effect_per_venue_at_t_conf20")},
                     ensure_ascii=False, indent=2))
    print("\n--- per venue (reference config, 累积口径) ---")
    for L, d in pv.items():
        print(f"  {L:<10} eps={d['n_episodes_of_venue']:<3} pos={d['n']:<3} "
              f"conf={d['lvl2_ok']}/{d['lvl2_wrong']} "
              f"prov_only={d['lvl1_ok']}/{d['lvl1_wrong']} "
              f"rej={d['rej_no_cand']}+{d['rej_below']} "
              f"inl_p50={d['max_inl_p50']} run_p50={d['run_p50']}")
    if args.json_out:
        args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"\n[saved] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
