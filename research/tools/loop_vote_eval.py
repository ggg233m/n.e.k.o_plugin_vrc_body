"""三级确认协议评测：candidate / provisional_loop / confirmed_loop。

协议（用户 2026-09-20 定稿）
--------------------------
| 级别 | 条件 | 允许做什么 |
|---|---|---|
| ``candidate`` | BoW Top-20 命中 | 只表示值得检查 |
| ``provisional_loop`` | ORB 几何证据存在 | 允许写入**候选回环记录**，**不合并地点节点** |
| ``confirmed_loop`` | 因果多帧支持（连续支持长度 ≥ N） | 才允许**合并地点节点** |

因果多帧投票
------------
- 只用**当前帧及其过去帧**（lag ≥ 0），**不允许未来帧**
- 支持长度 = 从最近滞后 (lag₁) 起、向外**连续**通过 ``T_vote`` 的对数
- 对角匹配 (i−lag, j−lag) 隐含同一行进偏移 → **多帧投票与"路线历史一致"是同一件事**，
  不需要额外朝向积分

评测要求（用户明确列出，逐条对应）
--------------------------------
- 报告 接受且正确 / 接受但错误 / 拒绝 **三类**，**不把"全部拒绝"计作成功**
- **每个地点分别统计**（dancepool / hall / pooldeck / statue），不被总体平均掩盖
- **leave-one-loop-out**：在 3 个地点上选参，留出第 4 个测试
- 不只报平均召回

用法
----
  .venv/Scripts/python.exe research/tools/loop_vote_eval.py \
      --data .tmp/loop_vote_pairs.npz --json-out .tmp/loop_vote_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
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
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--t-prov", type=int, default=25, help="provisional 内点门槛")
    ap.add_argument("--t-conf", type=int, default=40, help="confirmed 当前帧内点门槛")
    ap.add_argument("--t-vote", type=int, default=25, help="confirmed 投票帧内点门槛")
    ap.add_argument("--n-vote", type=int, default=2, help="confirmed 连续支持长度门槛")
    ap.add_argument("--need-same-seg", action="store_true",
                    help="要求支持帧与当前帧同属一次访问（路线连续性）")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    qidx = d["query_idx"].astype(int)
    CAND = d["cand"]
    INL = d["inliers"].astype(np.int32)
    CEL = d["cells"].astype(np.int32)
    DOM = d["dominant"].astype(np.float32)
    REM = d["remaining"].astype(np.int32)
    VOK = d["vote_ok"].astype(bool)
    VSAME = d["vote_same_segment"].astype(bool)
    lags = [int(x) for x in d["lags"]]
    times = d["times"].astype(float)
    label = np.array([None if str(x) == "None" else str(x) for x in d["label"]],
                     dtype=object)
    min_sep = float(d["min_sep"])
    nl = len(lags)
    print(f"[info] 正查询={len(qidx)}  topk={CAND.shape[1]}  lags={lags}")

    def earlier(i):
        L = label[i]
        if L is None:
            return []
        return [(a, b) for a, b in gt_segments_of(L) if b <= times[i] - min_sep]

    GT = np.zeros(CAND.shape, dtype=bool)
    for r, i in enumerate(qidx):
        segs = earlier(int(i))
        if not segs:
            continue
        for c, j in enumerate(CAND[r]):
            if j < 0:
                continue
            GT[r, c] = any(a - TOL <= times[j] <= b + TOL for a, b in segs)

    # ---------------- 特征：连续支持长度 ----------------
    def support_len(r, c, t_vote, need_same):
        """从最近滞后向外连续通过 t_vote 的个数。lag0 不计入（它单独由 t_conf 判）。"""
        n = 0
        for li in range(1, nl):
            if not VOK[r, c, li]:
                break
            if need_same and not VSAME[r, c, li]:
                break
            if INL[r, c, li] < t_vote:
                break
            n += 1
        return n

    SUP = np.zeros(CAND.shape, dtype=np.int8)
    for r in range(CAND.shape[0]):
        for c in range(CAND.shape[1]):
            if CAND[r, c] < 0:
                continue
            SUP[r, c] = support_len(r, c, args.t_vote, args.need_same_seg)

    # ---------------- 特征分离度诊断 ----------------
    gt0 = GT & (CAND >= 0)
    ng0 = (~GT) & (CAND >= 0)
    diag = {
        "lag0_inliers": {"gt_best": pct([INL[r, np.where(GT[r])[0]].max()
                                         for r in range(len(qidx)) if GT[r].any()]),
                         "non_gt_best": pct([INL[r, np.where(ng0[r])[0]].max()
                                             for r in range(len(qidx)) if ng0[r].any()])},
        "support_len": {
            "gt_best": pct([SUP[r, GT[r]].max() for r in range(len(qidx)) if GT[r].any()]),
            "non_gt_best": pct([SUP[r, ng0[r]].max() for r in range(len(qidx)) if ng0[r].any()]),
            "gt_hist": {str(k): int((SUP[gt0] == k).sum()) for k in range(nl)},
            "non_gt_hist": {str(k): int((SUP[ng0] == k).sum()) for k in range(nl)}},
        "lag0_cells": {"gt_best": pct([CEL[r, GT[r], 0].max()
                                       for r in range(len(qidx)) if GT[r].any()])},
        "lag0_remaining_after_biggest_cluster": {
            "gt_best": pct([REM[r, GT[r], 0].max()
                            for r in range(len(qidx)) if GT[r].any()]),
            "non_gt_best": pct([REM[r, ng0[r], 0].max()
                                for r in range(len(qidx)) if ng0[r].any()])},
        "lag0_dominant_cell_frac": {
            "gt_best": pct([DOM[r, GT[r], 0].min()
                            for r in range(len(qidx)) if GT[r].any()]),
            "non_gt_best": pct([DOM[r, ng0[r], 0].min()
                                for r in range(len(qidx)) if ng0[r].any()])},
    }

    # ---------------- 三级判定 ----------------
    def classify(i, t_prov, t_conf, n_vote, need_same):
        """返回 (level_reached, correct_or_None)。level: 0候选1暂时2确认。"""
        r = int(np.where(qidx == i)[0][0])
        cands = np.where(CAND[r] >= 0)[0]
        if len(cands) == 0:
            return 0, None
        # L3：先找满足 confirmed 的候选，取 lag0 内点最大者
        ok3 = [c for c in cands
               if INL[r, c, 0] >= t_conf
               and support_len(r, c, args.t_vote, need_same) >= n_vote]
        if ok3:
            c = max(ok3, key=lambda c: INL[r, c, 0])
            return 2, bool(GT[r, c])
        # L2：几何证据存在
        ok2 = [c for c in cands if INL[r, c, 0] >= t_prov]
        if ok2:
            c = max(ok2, key=lambda c: INL[r, c, 0])
            return 1, bool(GT[r, c])
        return 0, None

    def evaluate(rows_wanted, t_prov, t_conf, n_vote, need_same, label_fn=None):
        acc = {"reached_candidate": 0, "prov_ok": 0, "prov_wrong": 0,
               "conf_ok": 0, "conf_wrong": 0, "rejected_at_L2": 0,
               "no_candidate_at_all": 0}
        per = {}
        for i in rows_wanted:
            venue = label_fn(i) if label_fn else label[i]
            s = per.setdefault(venue, {"n": 0, "prov_ok": 0, "prov_wrong": 0,
                                       "conf_ok": 0, "conf_wrong": 0, "rejected": 0})
            s["n"] += 1
            lvl, ok = classify(i, t_prov, t_conf, n_vote, need_same)
            if lvl == 0:
                acc["no_candidate_at_all"] += 1
                s["rejected"] += 1
            elif lvl == 1:
                acc["reached_candidate"] += 1
                if ok:
                    acc["prov_ok"] += 1
                    s["prov_ok"] += 1
                else:
                    acc["prov_wrong"] += 1
                    s["prov_wrong"] += 1
                acc["rejected_at_L2"] += 1
                s["rejected"] += 1
            else:
                acc["reached_candidate"] += 1
                if ok:
                    acc["conf_ok"] += 1
                    s["conf_ok"] += 1
                else:
                    acc["conf_wrong"] += 1
                    s["conf_wrong"] += 1
        n = len(rows_wanted)
        acc["n"] = n
        acc["confirmed_recall"] = round(acc["conf_ok"] / n, 4) if n else None
        acc["confirmed_wrong_rate"] = round(acc["conf_wrong"] / n, 4) if n else None
        acc["provisional_recall"] = round((acc["prov_ok"] + acc["conf_ok"]) / n, 4) if n else None
        acc["provisional_wrong_rate"] = round(
            (acc["prov_wrong"] + acc["conf_wrong"]) / n, 4) if n else None
        acc["per_venue"] = {
            k: {"n": v["n"],
                "confirmed_recall": round(v["conf_ok"] / v["n"], 4),
                "confirmed_wrong": round(v["conf_wrong"] / v["n"], 4),
                "provisional_recall": round((v["prov_ok"] + v["conf_ok"]) / v["n"], 4),
                "provisional_wrong": round((v["prov_wrong"] + v["conf_wrong"]) / v["n"], 4),
                "rejected": v["rejected"]}
            for k, v in sorted(per.items())}
        return acc

    rows = [int(i) for i in qidx]
    venues = sorted({label[i] for i in rows})

    res: dict = {
        "protocol": "candidate -> provisional_loop -> confirmed_loop",
        "causal_vote": {"lags": lags, "lags_seconds_at_20fps":
                        [round(x / 20.0, 2) for x in lags],
                        "note": "lag 只用过去帧；对角匹配 = 常偏移一致性 = 路线历史一致"},
        "n_pos_queries": len(rows),
        "reference_config": {"t_prov": args.t_prov, "t_conf": args.t_conf,
                             "t_vote": args.t_vote, "n_vote": args.n_vote,
                             "need_same_segment": args.need_same_seg},
        "feature_separation": diag,
        "reference": evaluate(rows, args.t_prov, args.t_conf, args.n_vote,
                              args.need_same_seg),
    }

    # 多帧投票是否真的有用：单帧 vs 多帧
    cmp_ = {}
    for nv in (0, 1, 2, 3):
        cmp_[f"n_vote={nv}"] = evaluate(rows, args.t_prov, args.t_conf, nv,
                                        args.need_same_seg)
    res["singleframe_vs_multiframe"] = {
        k: {"confirmed_recall": v["confirmed_recall"],
            "confirmed_wrong_rate": v["confirmed_wrong_rate"],
            "conf_ok": v["conf_ok"], "conf_wrong": v["conf_wrong"],
            "rejected": v["no_candidate_at_all"] + v["rejected_at_L2"]}
        for k, v in cmp_.items()}

    # ---- leave-one-loop-out ----
    GRID = [(tc, tv, nv) for tc in (20, 30, 40, 60, 80)
            for tv in (15, 20, 30, 40) for nv in (1, 2, 3) if tv <= tc]
    lolo = {}
    for held in venues:
        tr = [i for i in rows if label[i] != held]
        te = [i for i in rows if label[i] == held]
        best = None
        for (tc, tv, nv) in GRID:
            e = evaluate(tr, args.t_prov, tc, nv, args.need_same_seg)
            if e["conf_wrong"] > 0:
                continue
            key = (e["conf_ok"], -nv, -tc)
            if best is None or key > best[0]:
                best = (key, (tc, tv, nv), e)
        if best is None:
            lolo[held] = {"n": len(te), "note": "训练集上不存在 conf_wrong=0 的配置"}
            continue
        _, (tc, tv, nv), etr = best
        ete = evaluate(te, args.t_prov, tc, nv, args.need_same_seg)
        lolo[held] = {
            "n": len(te), "chosen": {"t_conf": tc, "t_vote": tv, "n_vote": nv},
            "train": {"n": etr["n"], "confirmed_recall": etr["confirmed_recall"],
                      "confirmed_wrong": etr["confirmed_wrong_rate"]},
            "held_out": {"confirmed_recall": ete["confirmed_recall"],
                         "confirmed_wrong": ete["confirmed_wrong_rate"],
                         "confirmed_ok": ete["conf_ok"], "confirmed_wrong_n": ete["conf_wrong"],
                         "provisional_recall": ete["provisional_recall"]},
        }
    res["leave_one_loop_out"] = lolo
    res["note_lolo"] = ("选参目标 = 训练地点上 conf_wrong=0 时 maximization conf_ok；"
                       "held_out 用同一配置直接套用，不做任何回调。")

    # ---- 网格（参考点附近）供人工选工作点 ----
    grid = []
    for tc in (20, 30, 40, 60, 80):
        for nv in (0, 1, 2, 3):
            e = evaluate(rows, args.t_prov, tc, nv, args.need_same_seg)
            grid.append({"t_conf": tc, "n_vote": nv,
                         "confirmed_recall": e["confirmed_recall"],
                         "confirmed_wrong_rate": e["confirmed_wrong_rate"],
                         "rejected": e["no_candidate_at_all"] + e["rejected_at_L2"]})
    res["confirmed_grid_t_prov%d_t_vote%d" % (args.t_prov, args.t_vote)] = grid

    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"[saved] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
