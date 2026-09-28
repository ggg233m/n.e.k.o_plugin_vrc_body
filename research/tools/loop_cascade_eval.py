"""回环候选评测（读 loop_candidate_probe.py 的 npz，改阈值不重跑匹配）。

v2 修正（2026-09-20，用户复核发现的两处会改变结论的问题）

1. **"全部拒绝仍计成功"**：v1 在没有任何候选通过门控时，对全 -1 的数组执行
   ``argmax`` 仍返回第 0 个候选，若它恰好是真值就被算成命中 → 召回被高估。
   v2 明确区分**接受且正确 / 接受但错误 / 拒绝**三类。
2. **候选可以来自未来**：v1 允许匹配未来帧，这不是在线回环检测。
   v2 用 ``--online`` 只允许候选来自过去（``t_cand <= t_query - min_sep``）。
3. **覆盖率用整数格数**：3×3 网格下 5/9 = 0.5556，浮点阈值 0.56 实际要求 ≥6 格，
   四舍五入会悄悄改变门槛 → v2 一律用整数格数（cells）。

用法
----
  .venv/Scripts/python.exe research/tools/loop_cascade_eval.py --data .tmp/loop_candidates.npz
  .venv/Scripts/python.exe research/tools/loop_cascade_eval.py --data .tmp/loop_candidates.npz --online
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
GRID = 3  # 空间覆盖用的网格边长（3x3）


def _is_correct(ti: float, tb: float, label: str) -> bool:
    for a, b in gt_segments_of(label):
        if a - TOL <= ti <= b + TOL:
            continue
        if a - TOL <= tb <= b + TOL:
            return True
    return False


def build_rows(z, min_sep: float, online: bool) -> tuple[list, list]:
    times = z["times"].astype(float)
    qi = z["query_idx"].astype(int)
    INL = z["inliers"]
    COV, DOM, REM = z["coverage"], z["dominant"], z["remaining"]
    C = z["color"]
    N = len(times)

    pos, neg = [], []
    for a, i in enumerate(qi):
        ti = times[i]
        lab = gt_label(ti)
        if lab is None:
            continue
        if online:
            valid = [j for j in range(N)
                     if times[j] <= ti - min_sep]  # 只允许过去
        else:
            valid = [j for j in range(N) if abs(times[j] - ti) >= min_sep]
        if not valid:
            continue
        valid = np.array(valid)
        segs = gt_segments_of(lab)
        if online:
            # 在线口径下，只有"已经去过、现在又遇到"才算正样本：
            # 存在另一个同标签段、且它整体位于过去（b <= ti - min_sep）。
            earlier = [(a, b) for a, b in segs if b <= ti - min_sep]
            rev = bool(earlier)
            hit = np.array([rev and any(a - TOL <= t <= b + TOL
                                        for a, b in earlier)
                            for t in times[valid]])
        else:
            rev = len(segs) >= 2
            hit = np.array([rev and _is_correct(ti, t, lab)
                            for t in times[valid]])
        r = {
            "a": a, "t": ti, "label": lab, "revisited": rev,
            "hit": hit, "inl": INL[a, valid],
            "cells": np.round(COV[a, valid] * (GRID * GRID)).astype(int),
            "dom": DOM[a, valid], "rem": REM[a, valid],
            "colord": 1.0 - (C[i] @ C[valid].T),
        }
        (pos if rev else neg).append(r)
    return pos, neg


def accept(r, t_inl: int, cells: int, dom: float, rem: int):
    """返回被接受候选的索引，或 None（全部拒绝）。"""
    m = ((r["inl"] >= t_inl) & (r["cells"] >= cells)
         & (r["dom"] <= dom) & (r["rem"] >= rem))
    if not m.any():
        return None
    return int(np.argmax(np.where(m, r["inl"], -1)))


def eval_gate(pos, neg, t_inl, cells, dom=0.55, rem=12) -> dict:
    ok = wrong = rej = 0
    for r in pos:
        k = accept(r, t_inl, cells, dom, rem)
        if k is None:
            rej += 1
        elif r["hit"][k]:
            ok += 1
        else:
            wrong += 1
    n_fa = 0
    for r in neg:
        if accept(r, t_inl, cells, dom, rem) is not None:
            n_fa += 1
    n_pos, n_neg = len(pos), len(neg)
    return {
        "t_inl": t_inl, "cells": cells,
        "n_pos": n_pos, "n_neg": n_neg,
        "accepted_correct": ok, "accepted_wrong": wrong, "rejected": rej,
        "recall": round(ok / n_pos, 4) if n_pos else None,
        "wrong_accept_rate": round(wrong / n_pos, 4) if n_pos else None,
        "false_accept_rate": round(n_fa / n_neg, 4) if n_neg else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--min-sep", type=float, default=8.0)
    ap.add_argument("--online", action="store_true",
                    help="候选只允许来自过去（真正的在线回环检测）")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    z = np.load(args.data)
    pos, neg = build_rows(z, args.min_sep, args.online)
    print(f"[info] mode={'online(仅过去)' if args.online else 'offline(全片)'}"
          f"  pos={len(pos)} neg={len(neg)}")

    # 1. 检索阶段：不涉及门控，只看候选集里有没有真值
    retr = {}
    for K in (1, 5, 10, 20):
        ok = [bool(r["hit"][np.argsort(-r["inl"])[:K]].any()) for r in pos]
        retr[f"orb_recall@{K}"] = round(float(np.mean(ok)), 4) if pos else None
        okc = [bool(r["hit"][np.argsort(r["colord"])[:K]].any()) for r in pos]
        retr[f"color_recall@{K}"] = round(float(np.mean(okc)), 4) if pos else None

    # 2. 门控扫描（整数格数）
    grid = []
    for t_inl in (0, 25, 60, 105):
        for cells in (0, 5, 6, 7):
            grid.append(eval_gate(pos, neg, t_inl, cells))

    # 3. leave-one-loop-out：只在正样本上按 (recall - wrong_accept - false_accept) 选阈值
    labels = sorted({r["label"] for r in pos})
    lolo = {}
    for held in labels:
        tr = [r for r in pos if r["label"] != held]
        te = [r for r in pos if r["label"] == held]
        if not tr or not te:
            continue
        best, bsc = None, -1e9
        for t_inl in range(0, 400, 10):
            g = eval_gate(tr, neg, t_inl, 0)
            sc = (g["recall"] or 0) - (g["wrong_accept_rate"] or 0) \
                - (g["false_accept_rate"] or 0)
            if sc > bsc:
                best, bsc = t_inl, sc
        gh = eval_gate(te, neg, best, 0)
        lolo[held] = {"tuned_t_inl": int(best), "n_test": len(te),
                      "heldout_recall": gh["recall"],
                      "heldout_wrong_accept": gh["wrong_accept_rate"],
                      "heldout_rejected": gh["rejected"]}

    per = {}
    for lab in labels:
        rs = [r for r in pos if r["label"] == lab]
        g = eval_gate(rs, [], 0, 0)
        per[lab] = {"n": len(rs),
                    "top1_correct": g["accepted_correct"],
                    "top1_wrong": g["accepted_wrong"]}

    out = {"mode": "online" if args.online else "offline",
           "retrieval": retr, "gate_scan": grid,
           "leave_one_loop_out": lolo, "per_label": per}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"[saved] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
