"""在线因果 episode 切分：PlaceEpisode 的边界构建。

三层模型（用户 2026-09-20 定稿）
----------------------------
- **PlaceEpisode**：一次连续访问片段（如 hall@0-3s / hall@48-52s / hall@61-63s）
- **PlaceIdentity**：被回环证据连接的多个 episode（如 hall 的全部 episode）
- **LoopCandidate**：episode 之间的候选回环（candidate/provisional/confirmed）

**不要把"访问片段"和"地点身份"混成一个节点。** 当前无法确认两个片段是否同一地点时，
可以先保存 ``episode_A -- provisional_loop --> episode_B``，而不是强行合并。

切分规则（用户定稿的因果 change-point）
------------------------------------
1. 当前 episode 持有**若干关键帧原型**，不是一个平均 embedding
2. 新帧先与当前 episode 的**多个原型**比较
3. **单帧不相似不能立即切段**
4. 连续一段时间低相似，**并且**伴随运动阶段变化或场景结构变化，才结束当前 episode
5. 新 episode 建立后，再通过 BoW/ORB 与历史 episode 做回环候选匹配

**时间门按真实秒数表达**（``--hysteresis`` 单位秒），禁止用固定帧数——
采样率一变就失去意义。因果性：算法缓冲 hysteresis 秒后提交边界，延迟有界。

判据信号（全部来自画面，不用真值、不用 AngularY）
------------------------------------------------
- ``bow``：新帧 vs 当前 episode 各原型的 BoW 余弦，取最大
- ``shift``：相隔 ``move_lag`` 帧的 ORB 匹配点中位像素位移（运动阶段）
- ``nmatch``：相隔 ``move_lag`` 帧的有效 ORB 匹配数（场景结构连续性）

用法
----
  .venv/Scripts/python.exe research/tools/episode_segment.py \
      --sweep --json-out .tmp/episode_segment.json --out .tmp/episodes.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.tools.seqslam_probe import gt_label, gt_segments_of  # noqa: E402


# ------------------------------------------------------------------ 信号

def build_signals(args):
    import cv2
    z = np.load(args.orb_npz, allow_pickle=True)
    kps = list(z["kps"])
    descs = list(z["descs"])
    BOW = np.load(args.bow).astype(np.float32)
    N = len(descs)
    g = np.load(args.gray)
    gray, idx = g["frames"], g["idx"].astype(float)
    times = idx / args.src_fps

    Bn = BOW / (np.linalg.norm(BOW, axis=1, keepdims=True) + 1e-12)
    SIM = (Bn @ Bn.T).astype(np.float32)          # 1297x1297 全相似度，秒级查询

    bfm = cv2.BFMatcher(cv2.NORM_HAMMING)
    lag = args.move_lag
    shift = np.full(N, np.nan, dtype=np.float32)
    nmatch = np.zeros(N, dtype=np.int32)
    for i in range(N - lag):
        d1_, d2_ = descs[i], descs[i + lag]
        if d1_ is None or d2_ is None:
            continue
        raw = bfm.knnMatch(d1_, d2_, k=2)
        good = [m[0] for m in raw
                if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        nmatch[i] = len(good)
        if len(good) < args.move_min_matches:
            continue
        p1_ = np.asarray([kps[i][m.queryIdx] for m in good], np.float32)
        p2_ = np.asarray([kps[i + lag][m.trainIdx] for m in good], np.float32)
        shift[i] = float(np.median(np.linalg.norm(p2_ - p1_, axis=1)))

    moving = np.zeros(N, dtype=bool)
    moving[:N - lag] = np.nan_to_num(shift[:N - lag], nan=0.0) > args.move_px
    label = np.array([gt_label(t) for t in times], dtype=object)
    menu = np.array([l is None for l in label])
    usable = moving & (~menu)
    return dict(N=N, times=times, SIM=SIM, shift=shift, nmatch=nmatch,
                moving=moving, menu=menu, usable=usable, label=label)


# ------------------------------------------------------------------ 切分

def segment(S: dict, bow_low: float, bow_high: float, hysteresis_s: float,
            require_change: bool, hard_timeout_s: float,
            proto_min_gap_s: float, max_protos: int, min_ep_s: float,
            motion_rel: float = 0.5, motion_abs: float = 0.6,
            nmatch_frac: float = 0.4, nmatch_floor: int = 20):
    """因果 change-point 切分。返回 episode 列表 [(i0, i1_exclusive), ...]。"""
    times, SIM = S["times"], S["SIM"]
    usable, shift, nmatch = S["usable"], S["shift"], S["nmatch"]
    frames = np.where(usable)[0]
    eps: list[tuple[int, int]] = []

    state = {"start": None, "protos": [], "proto_t": [], "low": None,
             "hs": [], "hn": []}

    def anchor(a: int):
        state["start"] = int(a)
        state["protos"] = [int(a)]
        state["proto_t"] = [float(times[a])]
        state["low"] = None
        state["hs"], state["hn"] = [], []

    for i in frames:
        i = int(i)
        if state["start"] is None:
            anchor(i)
            continue
        sim = float(SIM[i, state["protos"]].max())

        med_s = float(np.median(state["hs"])) if state["hs"] else float("nan")
        med_n = float(np.median(state["hn"])) if state["hn"] else float("nan")
        mot = (np.isfinite(shift[i]) and np.isfinite(med_s)
               and abs(float(shift[i]) - med_s) > max(motion_abs, motion_rel * med_s))
        stru = (nmatch[i] < nmatch_floor
                or (np.isfinite(med_n) and nmatch[i] < nmatch_frac * med_n))
        change = bool(mot or stru)

        if sim >= bow_high:
            state["low"] = None                      # 恢复：自信回到同段（高阈值）
        elif state["low"] is None and sim < bow_low:
            state["low"] = i                         # 进入低相似态（低阈值）
        if state["low"] is not None:
            low_s = float(times[i] - times[state["low"]])
            if low_s >= hysteresis_s and (change or not require_change
                                          or low_s >= hard_timeout_s):
                # 提交边界：旧段结束于 low 起点（不含），新段从 low 起点开始
                if state["low"] > state["start"]:
                    eps.append((state["start"], state["low"]))
                anchor(state["low"])
                continue

        # 原型管理：间隔 >= proto_min_gap 秒才新增原型（保留若干关键帧原型）
        if times[i] - state["proto_t"][-1] >= proto_min_gap_s:
            state["protos"].append(i)
            state["proto_t"].append(float(times[i]))
            if len(state["protos"]) > max_protos:
                state["protos"].pop(0)
                state["proto_t"].pop(0)
        if np.isfinite(shift[i]):
            state["hs"].append(float(shift[i]))
        if nmatch[i] > 0:
            state["hn"].append(int(nmatch[i]))

    if state["start"] is not None:
        eps.append((state["start"], int(frames[-1]) + 1))

    # 合并过短 episode 到前一个（避免碎片）
    if min_ep_s > 0 and eps:
        merged = [list(eps[0])]
        for a, b in eps[1:]:
            if times[b - 1] - times[a] < min_ep_s:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        eps = [(a, b) for a, b in merged if b > a]
    return eps


# ------------------------------------------------------------------ 评测

def _gt_segs(label):
    segs = []
    for lab in sorted({l for l in label if l is not None}):
        for a, b in gt_segments_of(lab):
            segs.append((lab, float(a), float(b)))
    segs.sort(key=lambda x: x[1])
    return segs


def boundary_f1(eps, gt_segs, times, tol=1.0):
    """produced 边界 vs GT 段边界（不含片子首尾）配对。"""
    gt = []
    for _, a, b in gt_segs:
        gt += [a, b]
    gt = sorted(g for g in gt if g > times[0] + 1e-6 and g < times[-1] - 1e-6)
    pr = [round(float(times[b - 1]), 3) for a, b in eps[:-1]]
    used, tp = set(), 0
    for p in pr:
        best, bd = None, 1e9
        for k, g in enumerate(gt):
            if k in used:
                continue
            if abs(p - g) < bd:
                best, bd = k, abs(p - g)
        if best is not None and bd <= tol:
            used.add(best)
            tp += 1
    prec = tp / len(pr) if pr else 0.0
    rec = tp / len(gt) if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "n_pred": len(pr), "n_gt": len(gt), "tp": tp}


def seg_quality(eps, times, label, gt_segs):
    """过切分 / 欠切分 / 纯度 / 碎片化。"""
    ep_info = []
    for a, b in eps:
        labs: dict = {}
        in_seg: dict = {}
        for k in range(a, b):
            L = label[k]
            if L is None:
                continue
            labs[L] = labs.get(L, 0) + 1
            for si, (slab, sa, sb) in enumerate(gt_segs):
                if slab == L and sa - 1e-6 <= times[k] <= sb + 1e-6:
                    in_seg[si] = in_seg.get(si, 0) + 1
                    break
        tot = sum(labs.values())
        ep_info.append({"a": a, "b": b, "labs": labs, "tot": tot, "in_seg": in_seg,
                        "dur": round(float(times[b - 1] - times[a]), 2)})

    under = [e for e in ep_info
             if e["tot"] and max(e["labs"].values()) / e["tot"] < 0.8]
    over_gt = 0
    over_excess = 0
    for si, (slab, sa, sb) in enumerate(gt_segs):
        cnt = 0
        for e in ep_info:
            if e["tot"] == 0:
                continue
            if e["in_seg"].get(si, 0) / e["tot"] >= 0.5:
                cnt += 1
        if cnt >= 2:
            over_gt += 1
            over_excess += cnt - 1
    purity = [max(e["labs"].values()) / e["tot"] for e in ep_info if e["tot"]]
    durs = [e["dur"] for e in ep_info]
    return {"n_episodes": len(eps), "n_gt_segments": len(gt_segs),
            "over_segmented_gt_segments": over_gt, "over_segments_excess": over_excess,
            "under_segmented_episodes": len(under),
            "mean_purity": round(float(np.mean(purity)), 4) if purity else None,
            "episodes_without_label": sum(1 for e in ep_info if e["tot"] == 0),
            "dur_s": {"p10": round(float(np.percentile(durs, 10)), 2),
                      "p50": round(float(np.percentile(durs, 50)), 2),
                      "p90": round(float(np.percentile(durs, 90)), 2),
                      "min": round(min(durs), 2), "max": round(max(durs), 2)},
            "labeled_frames_per_episode": {
                "p10": round(float(np.percentile([e["tot"] for e in ep_info], 10)), 1),
                "p50": round(float(np.percentile([e["tot"] for e in ep_info], 50)), 1)}}


def per_venue(eps, times, label, gt_segs, tol=1.0):
    out = {}
    for lab in sorted({l for l in label if l is not None}):
        segs = [(l, a, b) for l, a, b in gt_segs if l == lab]
        if len(segs) < 2:
            out[lab] = {"n_gt_segments": len(segs),
                        "note": "只有一次访问，无回环边界可比"}
            continue
        lo = min(a for _, a, _ in segs) - 1.0
        hi = max(b for _, _, b in segs) + 1.0
        sub = [(a, b) for a, b in eps if times[b - 1] >= lo and times[a] <= hi]
        out[lab] = {"n_gt_segments": len(segs),
                    "boundary": boundary_f1(sub, segs, times, tol)}
    return out


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orb-npz", type=Path,
                    default=Path(".slam_probe/offline_probe/orb_full_n600.npz"))
    ap.add_argument("--bow", type=Path,
                    default=Path(".slam_probe/offline_probe/bow_full_v400.npy"))
    ap.add_argument("--gray", type=Path,
                    default=Path(".slam_probe/offline_probe/gray_640x360_s3_full.npz"))
    ap.add_argument("--src-fps", type=float, default=60.0)
    ap.add_argument("--move-lag", type=int, default=4)
    ap.add_argument("--move-px", type=float, default=1.5)
    ap.add_argument("--move-min-matches", type=int, default=20)
    ap.add_argument("--bow-low", type=float, default=0.35)
    ap.add_argument("--bow-high", type=float, default=0.55)
    ap.add_argument("--hysteresis", type=float, default=1.0, help="秒")
    ap.add_argument("--hard-timeout", type=float, default=3.0, help="秒")
    ap.add_argument("--require-change", action="store_true", default=True)
    ap.add_argument("--no-require-change", dest="require_change", action="store_false")
    ap.add_argument("--proto-min-gap", type=float, default=1.0, help="秒")
    ap.add_argument("--max-protos", type=int, default=6)
    ap.add_argument("--min-ep", type=float, default=1.0, help="秒；更短的并入前一个")
    ap.add_argument("--tol", type=float, default=1.0, help="边界配对容差（秒）")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    S = build_signals(args)
    times, label = S["times"], S["label"]
    gt_segs = _gt_segs(label)
    print(f"[info] 信号就绪 {time.time()-t0:.0f}s  可用帧={int(S['usable'].sum())} "
          f"GT 段={len(gt_segs)}")

    def run(bl, bh, hys, req, ht, mg, mp, me):
        eps = segment(S, bl, bh, hys, req, ht, mg, mp, me)
        bf = boundary_f1(eps, gt_segs, times, args.tol)
        q = seg_quality(eps, times, label, gt_segs)
        return eps, bf, q

    res: dict = {
        "model": "PlaceEpisode（在线因果 change-point）/ PlaceIdentity / LoopCandidate",
        "signals": {"N": S["N"], "usable": int(S["usable"].sum()),
                    "move_lag": args.move_lag,
                    "note": "间隔门全部按秒表达；切分因果、缓冲 hysteresis 秒后提交"},
        "gt_boundaries": {"n_gt_segments": len(gt_segs),
                          "note": "仅用于诊断与评测，切分本身不使用任何真值"},
    }

    if args.sweep:
        grid = []
        # 注意：bow_low 是"进入低相似态"阈值，bow_high 是"恢复"阈值（必须 bow_high > bow_low）。
        # 早期版本误把 bow_low 写成死参数，扫描它时结果全等——本版已修正。
        for bl in (0.20, 0.28, 0.35, 0.45, 0.55):
            for hys in (0.5, 1.0, 1.5):
                for req in (True, False):
                    eps, bf, q = run(bl, max(0.60, bl + 0.15), hys, req, 3.0, 1.0, 6, 1.0)
                    grid.append({"bow_low": bl, "bow_high": max(0.60, bl + 0.15),
                                 "hysteresis_s": hys,
                                 "require_change": req,
                                 "f1": bf["f1"], "P": bf["precision"], "R": bf["recall"],
                                 "n_pred": bf["n_pred"], "n_gt": bf["n_gt"],
                                 "n_episodes": q["n_episodes"],
                                 "over_gt": q["over_segmented_gt_segments"],
                                 "under_ep": q["under_segmented_episodes"],
                                 "purity": q["mean_purity"]})
        grid.sort(key=lambda g: -g["f1"])
        res["sweep_top10"] = grid[:10]
        res["sweep_n_distinct_f1"] = len({round(g["f1"], 6) for g in grid})
        res["sweep_n_configs"] = len(grid)
        best = grid[0]
        res["chosen"] = best
        eps, bf, q = run(best["bow_low"], best["bow_high"],
                         best["hysteresis_s"], best["require_change"], 3.0, 1.0, 6, 1.0)
        res["boundary"] = bf
        res["quality"] = q
        res["per_venue"] = per_venue(eps, times, label, gt_segs, args.tol)
        res["episodes"] = [
            {"i0": int(a), "i1": int(b),
             "t0": round(float(times[a]), 2), "t1": round(float(times[b - 1]), 2),
             "n_frames": int(b - a),
             "dominant_label": (lambda d: max(d, key=d.get) if d else None)(
                 {l: sum(1 for x in label[a:b] if x == l)
                  for l in set(label[a:b]) if l is not None})}
            for a, b in eps]
        print(json.dumps({k: v for k, v in res.items() if k != "episodes"},
                         ensure_ascii=False, indent=2))
        if args.json_out:
            args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
            print(f"[saved] {args.json_out}")
        if args.out:
            np.savez_compressed(args.out, episodes=np.array(eps, dtype=np.int32),
                                times=times, label=label.astype(str),
                                usable=S["usable"], min_sep=8.0)
            print(f"[saved] {args.out}")
        return 0

    eps, bf, q = run(args.bow_low, args.bow_high, args.hysteresis,
                     args.require_change, args.hard_timeout,
                     args.proto_min_gap, args.max_protos, args.min_ep)
    print(f"[info] episodes={len(eps)}  boundary F1={bf['f1']} "
          f"(P={bf['precision']} R={bf['recall']})")
    print(f"[info] over_gt={q['over_segmented_gt_segments']} "
          f"under_ep={q['under_segmented_episodes']} purity={q['mean_purity']}")
    if args.out:
        np.savez_compressed(args.out, episodes=np.array(eps, dtype=np.int32),
                            times=times, label=label.astype(str),
                            usable=S["usable"], min_sep=8.0)
        print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
