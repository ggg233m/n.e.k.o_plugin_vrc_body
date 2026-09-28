"""确认门 ROC 复算：几何 + 空间覆盖门槛到底能不能确认真回环。

输入 ``--data`` = bow_loop_eval.py ``--dump-npz`` 落盘的
``(inliers, cells)`` 全量矩阵（内点数是**真实双向匹配**算出来的，不是代理指标）。

回答三个问题（每个都分开统计，避免总体平均数掩盖困难地点）
--------------------------------------------------------
1. **真值配对本身能不能过门**：对每个正查询，在"时间上落在更早同标签段内"
   的候选里取内点最多的那个（称 *best_gt*），它的内点数 / 覆盖格数是多少？
   如果 best_gt 的格数普遍 < 门槛，那不是"检索没找到"，而是**门本身把真回环拒了**。
2. **假配对会不会过门**：非真值候选里内点最多的（*best_ng*）是否也过门？
   它过了就意味着"接受但错误"的风险，而不只是"拒绝"。
3. **门槛扫描**：t_inl × cells 网格下，confirm_recall（best_gt 过门）与
   false_accept（best_ng 过门且内点更高）如何权衡。

用法
----
  .venv/Scripts/python.exe research/tools/loop_gate_sweep.py \
      --data .tmp/bow_pairs_online.npz \
      --bow .slam_probe/offline_probe/bow_full_v400.npy \
      --json-out .tmp/loop_gate_sweep.json
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

from research.tools.seqslam_probe import gt_label, gt_segments_of  # noqa: E402

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
    ap.add_argument("--bow", type=Path,
                    default=Path(".slam_probe/offline_probe/bow_full_v400.npy"))
    ap.add_argument("--src-fps", type=float, default=60.0)
    ap.add_argument("--min-sep", type=float, default=8.0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    qidx = d["query_idx"]
    times = d["times"].astype(float)
    label = np.array([str(x) for x in d["label"]], dtype=object)
    usable = d["usable"].astype(bool)
    INL = d["inliers"].astype(np.int32)
    CEL = d["cells"].astype(np.int32)
    N = len(times)
    lab = np.array([gt_label(t) for t in times], dtype=object)   # 与评测器同源
    BOW = np.load(args.bow).astype(np.float32)

    qpos = {int(i): r for r, i in enumerate(qidx)}

    def earlier(i):
        L = label[i]
        if L == "None" or not L:
            return []
        return [(a, b) for a, b in gt_segments_of(L) if b <= times[i] - args.min_sep]

    def is_gt(i, j):
        return any(a - TOL <= times[j] <= b + TOL for a, b in earlier(i))

    pos_queries = [int(i) for i in qidx if earlier(int(i))]
    print(f"[info] queries={len(qidx)}  在线正样本={len(pos_queries)}  N={N}")

    Bn = BOW / (np.linalg.norm(BOW, axis=1, keepdims=True) + 1e-12)

    # ---- 每个正查询：best_gt / best_ng 的内点数与覆盖格数 ----
    rows = []
    for i in pos_queries:
        r = qpos[i]
        cand = np.where(usable & (times <= times[i] - args.min_sep))[0]
        cand = cand[cand != i]
        if cand.size == 0:
            continue
        inl = INL[r, cand]
        cel = CEL[r, cand]
        gt_m = np.array([is_gt(i, int(j)) for j in cand])
        # BoW 排序（与评测器同一候选集）
        sim = Bn[cand] @ Bn[i]
        bow_order = cand[np.argsort(-sim)]
        rows.append({
            "i": i, "times": float(times[i]), "label": label[i],
            "cand": cand, "inl": inl, "cel": cel, "gt": gt_m,
            "bow_order": bow_order,
        })

    def best_of(rw, mask):
        if not mask.any():
            return None
        idx = np.where(mask)[0]
        k = idx[int(np.argmax(rw["inl"][idx]))]
        return {"j": int(rw["cand"][k]), "inl": int(rw["inl"][k]),
                "cells": int(rw["cel"][k]), "t": float(times[rw["cand"][k]])}

    gt_inl, gt_cel, ng_inl, ng_cel, ng_higher = [], [], [], [], 0
    for rw in rows:
        bg = best_of(rw, rw["gt"])
        bn = best_of(rw, ~rw["gt"])
        if bg:
            gt_inl.append(bg["inl"])
            gt_cel.append(bg["cells"])
        if bn:
            ng_inl.append(bn["inl"])
            ng_cel.append(bn["cells"])
        if bg and bn and bn["inl"] > bg["inl"]:
            ng_higher += 1

    out: dict = {
        "probe": "confirmation-gate ROC on真值配对（内点数=真实双向匹配）",
        "n_pos_queries": len(rows),
        "best_gt": {"inliers": pct(gt_inl), "cells": pct(gt_cel),
                    "cell_hist": {str(c): int((np.array(gt_cel) == c).sum())
                                  for c in range(0, 10)}},
        "best_non_gt": {"inliers": pct(ng_inl), "cells": pct(ng_cel),
                        "cell_hist": {str(c): int((np.array(ng_cel) == c).sum())
                                      for c in range(0, 10)}},
        "queries_where_non_gt_outranks_gt": ng_higher,
    }

    # ---- 门槛扫描：best_gt 过门率 vs best_non_gt 过门率 ----
    grid = []
    for t_inl in (10, 20, 30, 40, 60, 80, 105, 130):
        for cells in (0, 3, 4, 5, 6, 7):
            ok_gt = sum(1 for rw in rows
                        if (bg := best_of(rw, rw["gt"]))
                        and bg["inl"] >= t_inl and bg["cells"] >= cells)
            ok_ng = sum(1 for rw in rows
                        if (bn := best_of(rw, ~rw["gt"]))
                        and bn["inl"] >= t_inl and bn["cells"] >= cells)
            n = len(rows)
            grid.append({
                "t_inl": t_inl, "cells": cells,
                "gt_pass_rate": round(ok_gt / n, 4),
                "ng_pass_rate": round(ok_ng / n, 4),
                "bal": round((ok_gt / n + 1 - ok_ng / n) / 2, 4)})
    grid.sort(key=lambda g: -g["bal"])
    out["gate_grid_top12"] = grid[:12]
    out["note_grid"] = ("gt_pass_rate = 真值配对能过门的正查询比例（这是确认阶段的"
                        "召回上限）；ng_pass_rate = 非真值配对也能过门的正查询比例"
                        "（假接受风险，不是单访负查询的 FPR）。")

    # ---- 分级诊断：BoW Top-K 内，真值候选与门的交互 ----
    stage = {}
    for K in (5, 10, 20):
        gt_in_topk = 0
        gt_in_topk_pass = 0
        for rw in rows:
            top = list(rw["bow_order"][:K])
            ok = False
            for c in top:
                k = int(np.where(rw["cand"] == c)[0][0])
                if not rw["gt"][k]:
                    continue
                ok = True
                if rw["inl"][k] >= 30 and rw["cel"][k] >= 6:
                    gt_in_topk_pass += 1
                    break
            if ok:
                gt_in_topk += 1
        stage[f"top{K}"] = {
            "gt_candidate_in_topk": gt_in_topk,
            "gt_candidate_in_topk_rate": round(gt_in_topk / len(rows), 4),
            "of_which_pass_default_gate(inl>=30,cells>=6)": gt_in_topk_pass,
            "gate_kills": gt_in_topk - gt_in_topk_pass}
    out["stage_diagnosis_default_gate"] = stage

    # ---- 困难地点分开报 ----
    per_label = {}
    for rw in rows:
        t = per_label.setdefault(rw["label"], {"n": 0, "gt_inl": [], "gt_cel": []})
        bg = best_of(rw, rw["gt"])
        t["n"] += 1
        if bg:
            t["gt_inl"].append(bg["inl"])
            t["gt_cel"].append(bg["cells"])
    out["per_label"] = {
        k: {"n": v["n"], "best_gt_inliers": pct(v["gt_inl"]),
            "best_gt_cells": pct(v["gt_cel"]),
            "best_gt_cells_ge6": (round(float(np.mean(np.array(v["gt_cel"]) >= 6)), 4)
                                  if v["gt_cel"] else None)}
        for k, v in sorted(per_label.items())}

    # ---- 检索排序对比 + McNemar（同一批正查询，配对检验）----
    # 关键问题：BoW@10 差"1 个查询"到底是不是实质差距？n=38 时 1 个查询 = 2.6pp，
    # 单看百分比会过度解读，必须用配对检验。
    def hits_by(rank_fn, K):
        out_ = {}
        for rw in rows:
            top = list(rank_fn(rw)[:K])
            out_[rw["i"]] = any(rw["gt"][int(np.where(rw["cand"] == c)[0][0])]
                                for c in top)
        return out_

    def oracle_rank(rw):
        return rw["cand"][np.argsort(-rw["inl"])]

    def bow_rank(rw):
        return rw["bow_order"]

    rec = {}
    for K in (1, 5, 10, 20):
        for nm, fn in (("orb_oracle", oracle_rank), ("bow", bow_rank)):
            h = hits_by(fn, K)
            rec[f"{nm}@{K}"] = round(sum(h.values()) / len(h), 4) if h else None
    out["retrieval_on_same_queries"] = rec

    from math import comb

    def mcnemar(a: dict, b: dict, name: str):
        keys = sorted(set(a) & set(b))
        n01 = sum(1 for k in keys if a[k] and not b[k])   # A 中 B 不中
        n10 = sum(1 for k in keys if b[k] and not a[k])   # B 中 A 不中
        n = n01 + n10
        p = 1.0 if n == 0 else min(
            1.0, 2 * sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / (2 ** n))
        return {"A_only_hits": n01, "B_only_hits": n10,
                "discordant": n, "exact_binomial_two_sided_p": round(p, 4),
                "names": name}

    out["mcnemar"] = [
        mcnemar(hits_by(bow_rank, 10), hits_by(oracle_rank, 20),
                "bow@10 vs orb_oracle@20"),
        mcnemar(hits_by(bow_rank, 10), hits_by(oracle_rank, 10),
                "bow@10 vs orb_oracle@10"),
        mcnemar(hits_by(bow_rank, 20), hits_by(oracle_rank, 20),
                "bow@20 vs orb_oracle@20"),
    ]

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"[saved] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
