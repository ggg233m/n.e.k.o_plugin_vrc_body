"""三级确认协议 —— **偏移无关**的地点级多帧投票评测。

与 `loop_vote_eval.py` 的区别（这是关键）
---------------------------------------
| | 对角投票 `loop_vote_eval` | **地点级投票（本脚本）** |
|---|---|---|
| 投票对 | `(i−lag, j−lag)` | `(i−lag, 任意落在同一候选地点的帧)` |
| 隐含假设 | 两次访问**同向、同速、同序** | **无** |
| 与用户协议 | 不符 | 符合："累计在**候选地点**上的相邻帧连续支持长度" |

对角形式在本素材上失败有结构原因：重访是**反向/变序行进**，
`(i−lag, j−lag)` 要求偏移恒定，反向行进下偏移每帧变 −2。

三级协议（用户 2026-09-20 定稿）
------------------------------
- ``candidate``：BoW Top-20 命中 → 只表示值得检查
- ``provisional_loop``：ORB 几何证据存在 → 允许写候选回环记录，**不合并地点节点**
- ``confirmed_loop``：连续片段支持（相邻多帧各自独立匹配同一地点）→ 才允许**合并地点节点**

评测要求逐条对应
--------------
- 三类：接受且正确 / 接受但错误 / 拒绝；**"全部拒绝"单独计数，不计成功**
- 每个地点分别统计；**leave-one-loop-out**；不只报平均召回
- 额外报**行进方向诊断**：支持帧隐含的偏移随滞后如何变化 → 直接检验"反向行进"假设

用法
----
  .venv/Scripts/python.exe research/tools/loop_place_vote_eval.py \
      --index .tmp/loop_frame_index.npz --lags 0,10,20,30 \
      --json-out .tmp/loop_place_vote_eval.json
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

from research.tools.seqslam_probe import gt_segments_of  # noqa: E402

TOL = 1.0


def pct(a, ps=(10, 25, 50, 75, 90)):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {}
    return {f"p{p}": round(float(np.percentile(a, p)), 1) for p in ps} | {
        "min": round(float(a.min()), 1), "max": round(float(a.max()), 1),
        "mean": round(float(a.mean()), 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--query-stride", type=int, default=8)
    ap.add_argument("--lags", type=str, default="0,10,20,30")
    ap.add_argument("--t-prov", type=int, default=25)
    ap.add_argument("--t-conf", type=int, default=40)
    ap.add_argument("--t-vote", type=int, default=25)
    ap.add_argument("--n-vote", type=int, default=2)
    ap.add_argument("--need-same-visit", action="store_true",
                    help="要求支持帧与当前帧同属一次访问（路线连续性）")
    ap.add_argument("--place-mode", choices=("gt_segment", "temporal_cluster"),
                    default="temporal_cluster",
                    help="候选地点的定义。gt_segment=用真值段边界（**有信息泄漏**，仅作上界）；"
                         "temporal_cluster=只用候选帧的时间邻近性聚类（无泄漏，默认）")
    ap.add_argument("--cluster-gap", type=float, default=1.0,
                    help="temporal_cluster 模式下，候选帧间隔超过它即切成不同地点实例（秒）")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.index, allow_pickle=True)
    frames = d["frames"].astype(int)
    row_of = d["row_of"].astype(int)
    CAND = d["cand"]
    INL = d["inliers"].astype(np.int32)
    CEL = d["cells"].astype(np.int32)
    REM = d["remaining"].astype(np.int32)
    times = d["times"].astype(float)
    label = np.array([None if str(x) == "None" else str(x) for x in d["label"]],
                     dtype=object)
    usable = d["usable"].astype(bool)
    min_sep = float(d["min_sep"])
    K = int(d["topk"])
    N = len(times)
    lags = sorted({int(x) for x in args.lags.split(",")})
    print(f"[info] index frames={len(frames)} topk={K} lags={lags}")

    def earlier_segs(i):
        L = label[i]
        if L is None:
            return []
        return [(a, b) for a, b in gt_segments_of(L) if b <= times[i] - min_sep]

    queries = [i for i in range(0, N, args.query_stride) if usable[i]]
    pos = [i for i in queries if earlier_segs(i)]
    print(f"[info] queries={len(queries)}  在线正样本={len(pos)}")

    # ---------- 单帧证据查表：给帧 a、给**时间窗** w，返回最强证据 ----------
    def win_best(a, w):
        """返回 (inl, j, cells, rem)；无候选落在窗内返回 (-1, -1, 0, 0)。"""
        r = row_of[a]
        if r < 0:
            return -1, -1, 0, 0
        lo, hi = w
        best = (-1, -1, 0, 0)
        for c in range(K):
            j = int(CAND[r, c])
            if j < 0:
                continue
            if not (lo <= times[j] <= hi):
                continue
            v = int(INL[r, c])
            if v > best[0]:
                best = (v, j, int(CEL[r, c]), int(REM[r, c]))
        return best

    def gt_of_seg(seg):
        return (seg[0] - TOL, seg[1] + TOL)

    def places_of(i):
        """返回 [(window, is_gt_place)]。temporal_cluster 只用候选帧时间邻近性，不用真值。"""
        if args.place_mode == "gt_segment":
            return [(gt_of_seg(s), True) for s in earlier_segs(i)]
        r = row_of[i]
        if r < 0:
            return []
        js = sorted({int(CAND[r, c]) for c in range(K) if CAND[r, c] >= 0},
                    key=lambda j: times[j])
        wins, cur = [], []
        for j in js:
            if cur and times[j] - times[cur[-1]] > args.cluster_gap:
                wins.append(cur)
                cur = []
            cur.append(j)
        if cur:
            wins.append(cur)
        out = []
        for w in wins:
            lo, hi = times[w[0]] - 0.25, times[w[-1]] + 0.25
            is_gt = any(a - TOL <= times[j] <= b + TOL
                        for a, b in earlier_segs(i) for j in w)
            out.append(((lo, hi), is_gt))
        return out

    # ---------- 采样查询，构造每条查询的记录 ----------
    recs = []
    for i in pos:
        places = []
        for w, is_gt in places_of(i):
            ev = []
            for lag in lags:
                a = i - lag
                if a < 0 or not usable[a] or row_of[a] < 0:
                    ev.append(None)
                    continue
                if args.need_same_visit and label[a] != label[i]:
                    ev.append(None)
                    continue
                v, j, cel, rem = win_best(a, w)
                ev.append({"inl": v, "j": j, "cells": cel, "rem": rem, "a": a})
            places.append({"window": w, "is_gt": is_gt, "ev": ev})
        recs.append({"i": i, "label": label[i], "segs": places})
    by_i = {r["i"]: r for r in recs}

    # ---------- 连续支持长度（从 lag0 向外，不允许断档）----------
    def run_len(ev, t_vote, start=0):
        n = 0
        for x in ev[start:]:
            if x is None or x["inl"] < t_vote:
                break
            n += 1
        return n

    def pick(i):
        """在**所有**候选地点里选证据最强的那个（不使用真值）。"""
        r = by_i[i]
        best = None
        for cs in r["segs"]:
            ev = cs["ev"]
            if not ev or ev[0] is None:
                continue
            key = (ev[0]["inl"], run_len(ev, args.t_vote), round(cs["window"][0], 3))
            if best is None or key > best[0]:
                best = (key, ev, cs)
        return best

    # ---------- 方向诊断：偏移随滞后如何变化 ----------
    dirn = []
    for r in recs:
        b = pick(r["i"])
        if b is None:
            continue
        key, ev, cs = b
        d0 = ev[0]["j"] - r["i"]
        for li in range(1, len(lags)):
            if ev[li] is not None and ev[li]["inl"] >= args.t_vote:
                off_now = ev[li]["j"] - ev[li]["a"]
                dirn.append(off_now - d0)
    direction_diag = {
        "n_support_pairs": len(dirn),
        "delta_offset_frames": pct(dirn),
        "interpretation": ("≈0 → 同向同速（对角成立）；"
                          "≈+2·lag → 反向行进（对角失效，地点级才是对的）"),
        "hist": {},
    }
    if dirn:
        arr = np.asarray(dirn)
        for lo, hi, nm in ((-5, 5, "|Δ|<5 同向"), (5, 25, "Δ 5~25"),
                           (25, 55, "Δ 25~55 (=2·lag 反向)"), (55, 999, "Δ>55")):
            direction_diag["hist"][nm] = int(((arr >= lo) & (arr < hi)).sum())

    # ---------- 三级判定 ----------
    def classify(rec_i, t_prov, t_conf, n_vote):
        b = pick(rec_i)
        if b is None:
            return 0, None
        key, ev, cs = b
        run = run_len(ev, args.t_vote)
        if ev[0]["inl"] >= t_conf and run >= n_vote:
            return 2, ev[0]["j"]
        if ev[0]["inl"] >= t_prov:
            return 1, ev[0]["j"]
        return 0, None

    def gt_ok(i, j):
        if j < 0:
            return False
        return any(a - TOL <= times[j] <= b + TOL for a, b in earlier_segs(i))

    def evaluate(rows, t_prov, t_conf, n_vote):
        acc = {"n": len(rows), "no_candidate": 0, "prov_ok": 0, "prov_wrong": 0,
               "conf_ok": 0, "conf_wrong": 0}
        per = {}
        for i in rows:
            s = per.setdefault(label[i], {"n": 0, "conf_ok": 0, "conf_wrong": 0,
                                          "prov_ok": 0, "prov_wrong": 0,
                                          "rejected": 0})
            s["n"] += 1
            lvl, j = classify(i, t_prov, t_conf, n_vote)
            if lvl == 0:
                acc["no_candidate"] += 1
                s["rejected"] += 1
                continue
            ok = gt_ok(i, j)
            if lvl == 1:
                acc["prov_ok" if ok else "prov_wrong"] += 1
                s["prov_ok" if ok else "prov_wrong"] += 1
                s["rejected"] += 1
            else:
                acc["conf_ok" if ok else "conf_wrong"] += 1
                s["conf_ok" if ok else "conf_wrong"] += 1
        n = max(acc["n"], 1)
        acc["confirmed_recall"] = round(acc["conf_ok"] / n, 4)
        acc["confirmed_wrong_rate"] = round(acc["conf_wrong"] / n, 4)
        acc["rejected_total"] = acc["no_candidate"] + acc["prov_ok"] + acc["prov_wrong"]
        acc["provisional_recall"] = round((acc["prov_ok"] + acc["conf_ok"]) / n, 4)
        acc["provisional_wrong_rate"] = round((acc["prov_wrong"] + acc["conf_wrong"]) / n, 4)
        acc["per_venue"] = {
            k: {"n": v["n"],
                "confirmed_recall": round(v["conf_ok"] / v["n"], 4),
                "confirmed_wrong": round(v["conf_wrong"] / v["n"], 4),
                "provisional_recall": round((v["prov_ok"] + v["conf_ok"]) / v["n"], 4),
                "provisional_wrong": round((v["prov_wrong"] + v["conf_wrong"]) / v["n"], 4),
                "rejected": v["rejected"]}
            for k, v in sorted(per.items())}
        return acc

    rows = [r["i"] for r in recs]
    venues = sorted({label[i] for i in rows})
    res: dict = {
        "protocol": "candidate -> provisional_loop -> confirmed_loop (偏移无关地点级投票)",
        "vote_form": "place-level, offset-agnostic（与用户协议一致）",
        "n_pos_queries": len(rows),
        "lags": lags, "lags_seconds_at_20fps": [round(x / 20.0, 2) for x in lags],
        "place_mode": args.place_mode,
        "cluster_gap_s": args.cluster_gap if args.place_mode == "temporal_cluster" else None,
        "leakage_note": ("gt_segment 模式用真值段边界当候选地点 → **信息泄漏**，"
                        "wrong_accept 结构性恒为 0，只能当 recall 上界；"
                        "temporal_cluster 无泄漏，wrong_accept 才可测。"),
        "reference_config": {"t_prov": args.t_prov, "t_conf": args.t_conf,
                             "t_vote": args.t_vote, "n_vote": args.n_vote,
                             "need_same_visit": args.need_same_visit},
        "travel_direction_diagnostic": direction_diag,
        "reference": evaluate(rows, args.t_prov, args.t_conf, args.n_vote),
    }

    cmp_ = {}
    for nv in (0, 1, 2, 3, 4):
        if nv > len(lags):
            continue
        cmp_[f"n_vote={nv}"] = evaluate(rows, args.t_prov, args.t_conf, nv)
    res["singleframe_vs_multiframe"] = {
        k: {"confirmed_recall": v["confirmed_recall"],
            "confirmed_wrong_rate": v["confirmed_wrong_rate"],
            "conf_ok": v["conf_ok"], "conf_wrong": v["conf_wrong"],
            "rejected": v["rejected_total"]}
        for k, v in cmp_.items()}

    grid = []
    for tc in (20, 30, 40, 60, 80):
        for nv in range(0, len(lags) + 1):
            e = evaluate(rows, args.t_prov, tc, nv)
            grid.append({"t_conf": tc, "n_vote": nv,
                         "confirmed_recall": e["confirmed_recall"],
                         "confirmed_wrong_rate": e["confirmed_wrong_rate"],
                         "rejected": e["rejected_total"]})
    res["confirmed_grid"] = grid

    # ---------- leave-one-loop-out ----------
    # 选参规则：训练集上 conf_wrong=0 时最大化 conf_ok；**并列时优先更宽松的配置**
    # （更小的 n_vote / 更小的 t_conf）——否则会过拟合到最严的那档（实测会选到 n_vote=4）。
    GRID = [(tc, nv) for tc in (20, 30, 40, 60, 80) for nv in range(0, len(lags) + 1)]
    lolo = {}
    for held in venues:
        tr = [i for i in rows if label[i] != held]
        te = [i for i in rows if label[i] == held]
        best = None
        for (tc, nv) in GRID:
            e = evaluate(tr, args.t_prov, tc, nv)
            if e["conf_wrong"] > 0:
                continue
            key = (e["conf_ok"], -nv, -tc)      # 并列时偏好宽松
            if best is None or key > best[0]:
                best = (key, (tc, nv), e)
        if best is None:
            e0 = evaluate(tr, args.t_prov, args.t_conf, args.n_vote)
            lolo[held] = {"n": len(te),
                          "note": "训练集上不存在 conf_wrong=0 的配置",
                          "train_best_wrong": e0["conf_wrong"]}
            continue
        _, (tc, nv), etr = best
        ete = evaluate(te, args.t_prov, tc, nv)
        lolo[held] = {
            "n": len(te), "chosen": {"t_conf": tc, "n_vote": nv},
            "train": {"n": etr["n"], "confirmed_recall": etr["confirmed_recall"],
                      "confirmed_wrong": etr["confirmed_wrong_rate"]},
            "held_out": {"confirmed_recall": ete["confirmed_recall"],
                         "confirmed_wrong": ete["confirmed_wrong_rate"],
                         "conf_ok": ete["conf_ok"], "conf_wrong": ete["conf_wrong"],
                         "rejected": ete["rejected_total"],
                         "provisional_recall": ete["provisional_recall"]},
        }
    res["leave_one_loop_out"] = lolo

    # ---------- 固定配置的逐地点留出（不做调参，避免选参过拟合）----------
    fixed = {}
    for held in venues:
        te = [i for i in rows if label[i] == held]
        e = evaluate(te, args.t_prov, args.t_conf, args.n_vote)
        fixed[held] = {"n": len(te),
                       "confirmed_recall": e["confirmed_recall"],
                       "confirmed_wrong": e["confirmed_wrong_rate"],
                       "conf_ok": e["conf_ok"], "conf_wrong": e["conf_wrong"],
                       "provisional_recall": e["provisional_recall"],
                       "provisional_wrong": e["provisional_wrong_rate"],
                       "rejected": e["rejected_total"]}
    res["fixed_config_per_venue"] = {
        "config": {"t_prov": args.t_prov, "t_conf": args.t_conf,
                   "t_vote": args.t_vote, "n_vote": args.n_vote,
                   "note": "统一配置直接套到每个地点，不做任何按地点调参"},
        "per_venue": fixed}

    # ---------- 固定 t_conf=20，逐地点看 n_vote 的边际作用 ----------
    nv_per_venue = {}
    for held in venues:
        te = [i for i in rows if label[i] == held]
        nv_per_venue[held] = {
            f"n_vote={nv}": {"confirmed_recall": e["confirmed_recall"],
                             "confirmed_wrong": e["confirmed_wrong_rate"],
                             "conf_ok": e["conf_ok"], "conf_wrong": e["conf_wrong"]}
            for nv in range(0, len(lags) + 1)
            for e in [evaluate(te, args.t_prov, 20, nv)]}
    res["n_vote_effect_per_venue_at_t_conf20"] = nv_per_venue

    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"[saved] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
