"""语义地标**组合**通道（离线探针，不接实时链路）。

为什么要它
----------
`statue` 的 ORB 几何信号已三次独立实验确认不足（真值配对内点 p50 = 7~16，
低于任何可用门槛），所以它的下一通道不是"再调 ORB 阈值"，而是
**多帧稳定出现的区域组合**。同一个通道也用来消解 `pooldeck ↔ hall` 这类
视觉不可分辨的别名——它们几何上分不开，但**区域组合**可能分得开。

设计（严格遵守用户 2026-09-21 定稿的规格）
------------------------------------------
- 使用 **4x8 网格区域组合**，而不是整帧单标签；单个标签不足以成立。
- 每个 cell 的 token = (亮度相对档, 边缘密度档)；只有 **多帧稳定出现**
  （同一 token 在 episode 内出现比例 >= ``--stability``）的 cell 才进签名。
- 只有 **组合** 才算证据：要求 >= ``--min-cells`` 个 **结构化 cell**
  （边缘档 >= 2）在两个 episode 里同时稳定且 token 一致。
- 保存 **原始帧引用**（接近众数 token 的代表帧）+ 置信度。
- **只作为地点身份证据**：本工具不控制运动，失败时调用方保留 alias。
- 复用 `backend.avatar_identity._context_descriptor` 的 4x8 网格思路，
  但这里是灰度直接统计（素材只有灰度帧），不做人物剔除。

用法
----
  .venv/Scripts/python.exe research/tools/landmark_probe.py \
      --episodes .tmp/episodes.npz --pairs 34:5,34:6,34:23,39:23,39:30 \
      --json-out .tmp/landmark.json
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


ROWS, COLS = 4, 8


def frame_tokens(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把一帧压成 4x8=32 个 cell 的 (亮度相对档, 边缘密度档)。

    亮度档相对**本帧**均值取，抵消不同时刻的整体曝光差；
    边缘档相对**本帧**边缘中位数取，砍掉全局对比度的影响。
    这样 token 比较的是"区域内结构与周围区域的关系"，即组合关系。
    """
    import cv2

    h, w = img.shape[:2]
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(gx) + np.abs(gy)

    ys = (np.arange(ROWS + 1) * h // ROWS)
    xs = (np.arange(COLS + 1) * w // COLS)
    mean_tok = np.zeros((ROWS, COLS), np.int8)
    edge_tok = np.zeros((ROWS, COLS), np.int8)
    means = np.zeros((ROWS, COLS), np.float32)
    edges = np.zeros((ROWS, COLS), np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            blk = f[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            blk_m = mag[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            means[r, c] = float(blk.mean()) if blk.size else 0.0
            edges[r, c] = float(blk_m.mean()) if blk_m.size else 0.0

    fm = float(means.mean())
    fe = float(np.median(edges)) + 1e-3
    mean_tok = np.clip(np.rint((means - fm) / 25.0) + 2, 0, 4).astype(np.int8)
    ratio = edges / fe
    edge_tok = np.zeros((ROWS, COLS), np.int8)
    edge_tok[ratio > 0.70] = 1
    edge_tok[ratio > 1.15] = 2
    edge_tok[ratio > 1.70] = 3
    return mean_tok, edge_tok


def episode_signature(gray, frames, stability: float):
    """一个 episode 的区域组合签名（只保留多帧稳定的 cell）。

    返回 ``sig``（cell -> (mean_tok, edge_tok)）、``stable``（稳定性）、
    ``reps``（cell -> 代表帧号，取该 cell 最接近众数的那一帧）。
    """
    toks = []
    for i in frames:
        mt, et = frame_tokens(gray[i])
        toks.append(np.stack([mt, et], axis=-1))       # (R,C,2)
    if not toks:
        return {}, {}, {}, 0
    arr = np.stack(toks, axis=0)                        # (n,R,C,2)
    n = arr.shape[0]
    sig: dict[tuple[int, int], tuple[int, int]] = {}
    stab: dict[tuple[int, int], float] = {}
    reps: dict[tuple[int, int], int] = {}
    for r in range(ROWS):
        for c in range(COLS):
            cell = arr[:, r, c, :]                      # (n,2)
            keys, counts = np.unique(cell, axis=0, return_counts=True)
            k = int(np.argmax(counts))
            frac = float(counts[k]) / n
            if frac < stability:
                continue
            tok = (int(keys[k, 0]), int(keys[k, 1]))
            sig[(r, c)] = tok
            stab[(r, c)] = frac
            same = np.all(cell == keys[k], axis=1)
            idxs = np.flatnonzero(same)
            reps[(r, c)] = int(frames[int(idxs[len(idxs) // 2])])
    return sig, stab, reps, n


def landmark_match(a: dict, b: dict, min_cells: int, min_conf: float) -> dict:
    """区域组合一致性：要求 >= min_cells 个**结构化** cell 同时稳定且 token 一致。"""
    common = [cell for cell in a if cell in b and a[cell] == b[cell]]
    struct_a = [cell for cell, tok in a.items() if tok[1] >= 2]
    struct_b = [cell for cell, tok in b.items() if tok[1] >= 2]
    matched_struct = [c for c in common if a[c][1] >= 2]
    denom = max(1, min(len(struct_a), len(struct_b)))
    conf = len(matched_struct) / denom
    return {
        "matched_cells": len(common),
        "matched_structural": len(matched_struct),
        "structural_a": len(struct_a), "structural_b": len(struct_b),
        "confidence": round(conf, 4),
        "matched_cell_list": sorted(matched_struct),
        "confirms": bool(len(matched_struct) >= min_cells and conf >= min_conf),
    }


# ------------------------------------------------------------------ 视觉词组
# 为什么不用网格组合：实测（本素材）网格区域组合**判别力为零**（7 个同地点配对
# 与 7 个异地点配对的置信度全为 0.000，matched_structural 全为 0）。根因可诊断：
# 相机在平移，同一个地标在不同帧会落到不同网格位置，**网格下标根本不对应**；
# 再叠加"相对本帧均值"的归一化，token 变成了场景无关的自相似量。
# 因此改用**不需对应关系**的组合形式：同一帧内共现的 BoW 词对（视觉词组）。
# 词组天然是"组合关系"，且"多帧稳定共现"就是多帧稳定要求，不需要帧间对齐。

def episode_phrases(bow: np.ndarray, frames, top_per_frame: int,
                    min_frames: int, df_max: float, df: np.ndarray):
    """episode 的稳定视觉词组集合：同一帧内共现、且跨 >= min_frames 帧稳定。"""
    from collections import Counter

    counts: Counter = Counter()
    for f in frames:
        row = bow[f]
        ws = np.flatnonzero(row > 0)
        if ws.size == 0:
            continue
        order = ws[np.argsort(-row[ws])][:top_per_frame]
        keep = [int(w) for w in order if df[w] <= df_max]
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                counts[(keep[i], keep[j])] += 1
    return {p for p, c in counts.items() if c >= min_frames}, len(frames)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gray", type=Path,
                    default=Path(".slam_probe/offline_probe/gray_640x360_s3_full.npz"))
    ap.add_argument("--episodes", type=Path, default=Path(".tmp/episodes.npz"))
    ap.add_argument("--pairs", default="",
                    help="要评的组合 'q:p,q:p'（episode 行号）")
    ap.add_argument("--all-positive", action="store_true",
                    help="对每个阳性 query episode 评它全部更早 episode")
    ap.add_argument("--min-sep", type=float, default=8.0)
    ap.add_argument("--mode", default="phrase", choices=("grid", "phrase"),
                    help="grid=网格区域组合（已证判别力为零，保留作对照）；"
                         "phrase=同帧共现的 BoW 视觉词组（默认，不需帧间对应）")
    ap.add_argument("--bow", type=Path,
                    default=Path(".slam_probe/offline_probe/bow_full_v400.npy"))
    ap.add_argument("--stability", type=float, default=0.6,
                    help="grid 模式：cell token 在 episode 内出现的比例下限")
    ap.add_argument("--min-cells", type=int, default=3)
    ap.add_argument("--min-conf", type=float, default=0.40)
    ap.add_argument("--top-per-frame", type=int, default=8,
                    help="phrase 模式：每帧取权重最高的几个词参与组对")
    ap.add_argument("--min-phrases", type=int, default=3,
                    help="phrase 模式：判定'组合成立'所需的最少共享词组数")
    ap.add_argument("--phrase-stability", type=int, default=8,
                    help="phrase 模式：一个词组要在 episode 内至少共现多少帧才算稳定")
    ap.add_argument("--df-max", type=float, default=0.30,
                    help="phrase 模式：丢弃文档频率高于它的词（无信息高频词）")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    G = np.load(args.gray)
    gray = G["frames"]
    E = np.load(args.episodes, allow_pickle=True)
    eps = E["episodes"].astype(np.int64)
    label = E["label"].astype(str)
    times = E["times"].astype(np.float64)

    def dom(a, b):
        cnt: dict[str, int] = {}
        for i in range(int(a), int(b)):
            L = label[i]
            if L and L != "None":
                cnt[L] = cnt.get(L, 0) + 1
        return max(cnt, key=cnt.get) if cnt else None

    # 需要的 episode 集合
    pairs: list[tuple[int, int]] = []
    if args.pairs:
        for chunk in args.pairs.split(","):
            if chunk.strip():
                q, p = chunk.split(":")
                pairs.append((int(q), int(p)))
    if args.all_positive:
        for qi in range(len(eps)):
            a, b = eps[qi]
            L = dom(a, b)
            if L is None:
                continue
            for pi in range(qi):
                if times[eps[pi][0]] and times[eps[pi][1] - 1] + args.min_sep <= times[a] \
                        and dom(*eps[pi]) == L:
                    pairs.append((qi, pi))
    need = sorted({x for pr in pairs for x in pr})
    if not need:
        print("[error] 没有要评的组合，用 --pairs 或 --all-positive", file=sys.stderr)
        return 2

    BOW = None
    DF = None
    if args.mode == "phrase":
        BOW = np.load(args.bow).astype(np.float32)
        n_total = BOW.shape[0]
        DF = (BOW > 0).sum(axis=0).astype(np.float64) / max(1, n_total)

    sigs = {}
    for k in need:
        a, b = eps[k]
        frames = list(range(int(a), int(b)))
        win = [round(float(times[a]), 2), round(float(times[b - 1]), 2)]
        if args.mode == "phrase":
            phr, n = episode_phrases(BOW, frames, args.top_per_frame,
                                     args.phrase_stability, args.df_max, DF)
            sigs[k] = {"phrase": phr, "n": n, "window": win,
                       "dominant": dom(a, b)}
            print(f"[sig] ep{k:02d} {win} {sigs[k]['dominant']:<9} "
                  f"frames={n:<4} 稳定词组={len(phr):<5}")
        else:
            sig, stab, reps, n = episode_signature(gray, frames, args.stability)
            sigs[k] = {"sig": sig, "stab": stab, "reps": reps, "n": n,
                       "window": win, "dominant": dom(a, b)}
            print(f"[sig] ep{k:02d} {win} {sigs[k]['dominant']:<9} "
                  f"frames={n:<4} 稳定cell={len(sig):<3} "
                  f"结构化cell={sum(1 for t in sig.values() if t[1] >= 2)}")

    rows = []
    print("\n%-6s %-14s %-9s %-6s %-14s %-9s %5s %6s %6s %8s %s" % (
        "query", "window", "dom", "cand", "window", "dom",
        "match", "struct", "conf", "confirms", "GT同地点"))
    for qi, pi in pairs:
        if args.mode == "phrase":
            A, B = sigs[qi]["phrase"], sigs[pi]["phrase"]
            shared = A & B
            denom = max(1, min(len(A), len(B)))
            conf = len(shared) / denom
            m = {"matched_cells": len(shared), "matched_structural": len(shared),
                 "structural_a": len(A), "structural_b": len(B),
                 "confidence": round(conf, 4),
                 "matched_cell_list": [f"{w0}-{w1}" for w0, w1 in sorted(shared)],
                 "confirms": bool(len(shared) >= args.min_phrases
                                  and conf >= args.min_conf)}
        else:
            m = landmark_match(sigs[qi]["sig"], sigs[pi]["sig"],
                               args.min_cells, args.min_conf)
        same = sigs[qi]["dominant"] == sigs[pi]["dominant"]
        rows.append({"query": qi, "candidate": pi,
                     "query_window": sigs[qi]["window"],
                     "candidate_window": sigs[pi]["window"],
                     "query_dominant": sigs[qi]["dominant"],
                     "candidate_dominant": sigs[pi]["dominant"],
                     "gt_same_place": bool(same), **m})
        print("%-6s %-14s %-9s %-6s %-14s %-9s %5d %6d %6.3f %8s %s" % (
            "ep%02d" % qi, "%.2f-%.2f" % tuple(sigs[qi]["window"]),
            str(sigs[qi]["dominant"]),
            "ep%02d" % pi, "%.2f-%.2f" % tuple(sigs[pi]["window"]),
            str(sigs[pi]["dominant"]),
            m["matched_cells"], m["matched_structural"], m["confidence"],
            str(m["confirms"]), "是" if same else "否"))

    # 判别力统计
    pos = [r for r in rows if r["gt_same_place"]]
    neg = [r for r in rows if not r["gt_same_place"]]
    summary = {
        "n_pairs": len(rows), "n_same_place": len(pos), "n_diff_place": len(neg),
        "same_place_confirms": sum(1 for r in pos if r["confirms"]),
        "diff_place_confirms": sum(1 for r in neg if r["confirms"]),
        "same_place_conf_p50": (round(float(np.median([r["confidence"] for r in pos])), 4)
                                if pos else None),
        "diff_place_conf_p50": (round(float(np.median([r["confidence"] for r in neg])), 4)
                                if neg else None),
        "params": {"mode": args.mode, "stability": args.stability,
                   "min_cells": args.min_cells, "min_conf": args.min_conf,
                   "grid": f"{ROWS}x{COLS}", "top_per_frame": args.top_per_frame,
                   "min_phrases": args.min_phrases,
                   "phrase_stability": args.phrase_stability,
                   "df_max": args.df_max},
    }
    print("\n--- 判别力 ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 代表帧引用（原始帧号）与签名原文，供人工核验
    if args.mode == "phrase":
        sigs_out = {f"ep{k}": {"dominant": sigs[k]["dominant"],
                               "window": sigs[k]["window"],
                               "phrases": [f"{a}-{b}" for a, b in
                                           sorted(sigs[k]["phrase"])]}
                    for k in need}
    else:
        sigs_out = {f"ep{k}": {
            "dominant": sigs[k]["dominant"], "window": sigs[k]["window"],
            "structural_cell_frame_refs": {
                f"r{r}c{c}": sigs[k]["reps"][(r, c)]
                for (r, c) in sorted(sigs[k]["sig"])
                if sigs[k]["sig"][(r, c)][1] >= 2},
        } for k in need}
    res = {"summary": summary, "pairs": rows, "signatures": sigs_out,
           "note": "只作为地点身份证据；语义失败时调用方必须保留 alias，不合并"}
    if args.json_out:
        args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"[saved] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
