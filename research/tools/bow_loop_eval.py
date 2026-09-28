"""BoW 回环检索探针（same-sequence fitted retrieval probe）。

⚠️ **本次验证的性质**：词表 ``vocab_full_v400.npy`` 与 BoW ``bow_full_v400.npy``
来自**同一段素材**（`2026-09-18 07-31-11.mkv`）训练，因此结果**只能**用于：
验证候选检索接口 / 比较 Top-K 召回 / 比较候选数量 / 比较检索延迟 /
验证后续 ORB 几何确认能否正常接上。
**不能**据此宣称跨世界、跨录制或通用泛化。

口径（用户 2026-09-20 定稿）
--------------------------
- 查询：只查询**当前帧之前**的关键帧（``--allow-future`` 可切换为离线全片口径对照）
- 排除：时间上过近的帧（``--min-sep``）、**静止帧**、菜单遮挡帧
- 候选：BoW Top-5 / Top-10 / Top-20
- 确认：ORB 双向匹配 + 基础矩阵 RANSAC + 空间覆盖 + 多帧一致性
- 输出：检索 recall@K、接受且正确 / 接受但错误 / 拒绝、几何验证次数、p50/p95 延迟
- statue / pooldeck 单独报告，不被总体平均掩盖

验收目标（相对**同一帧集上**测出的全量 ORB 在线基线）
--------------------------------------------------
- BoW Top-10 召回 ≥ 全量 ORB 在线 Top-10 的 95%
- BoW Top-20 召回 ≥ 全量 ORB 在线 Top-20 的 95%

用法
----
  .venv/Scripts/python.exe research/tools/bow_loop_eval.py \
      --query-stride 20 --json-out .tmp/bow_loop_eval.json
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

TOL = 1.0
GRID = 3
GRID_N = GRID * GRID


# ---------------------------------------------------------------- 几何确认

def bidir_verify(kp1, d1, kp2, d2, ratio: float, ransac_px: float,
                 wh: tuple[float, float]):
    """双向匹配 + 基础矩阵 RANSAC。

    ``kp`` 为 ``(N,2) float32`` 像素坐标。返回 ``(inliers, cells, nq)``，
    其中 ``cells`` 是查询帧侧内点覆盖的 3x3 格数（0..9），``nq`` 为归一化坐标。
    """
    import cv2
    if d1 is None or d2 is None or len(d1) < 20 or len(d2) < 20:
        return 0, 0, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def one_way(a, b):
        raw = bf.knnMatch(a, b, k=2)
        return {m[0].queryIdx: m[0].trainIdx
                for m in raw if len(m) == 2 and m[0].distance < ratio * m[1].distance}

    fwd = one_way(d1, d2)
    bwd = one_way(d2, d1)
    mutual = [(qi, ti) for qi, ti in fwd.items() if bwd.get(ti) == qi]
    if len(mutual) < 8:
        return 0, 0, None
    p1 = np.asarray([kp1[qi] for qi, _ in mutual], dtype=np.float32)
    p2 = np.asarray([kp2[ti] for _, ti in mutual], dtype=np.float32)
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, ransac_px, 0.99)
    if F is None or mask is None:
        return 0, 0, None
    m = mask.ravel().astype(bool)
    inl = int(m.sum())
    if inl < 3:
        return inl, 0, None
    nq = p1[m] / np.asarray(wh, dtype=np.float32)
    cell = np.floor(np.clip(nq, 0.0, 0.999999) * GRID).astype(int)
    flat = cell[:, 0] * GRID + cell[:, 1]
    cells = int((np.bincount(flat, minlength=GRID_N) > 0).sum())
    return inl, cells, nq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orb-npz", type=Path,
                    default=Path(".slam_probe/offline_probe/orb_full_n600.npz"))
    ap.add_argument("--bow", type=Path,
                    default=Path(".slam_probe/offline_probe/bow_full_v400.npy"))
    ap.add_argument("--gray", type=Path,
                    default=Path(".slam_probe/offline_probe/gray_640x360_s3_full.npz"))
    ap.add_argument("--query-stride", type=int, default=20)
    ap.add_argument("--min-sep", type=float, default=8.0)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac-px", type=float, default=2.0)
    ap.add_argument("--move-lag", type=int, default=4, help="静止判据的帧间隔")
    ap.add_argument("--move-px", type=float, default=1.5,
                    help="匹配点中位位移 > 此值才算运动帧（overlap_new.py 同值）")
    ap.add_argument("--move-min-matches", type=int, default=20)
    ap.add_argument("--t-inl", type=int, default=30, help="确认阶段内点门槛")
    ap.add_argument("--cells", type=int, default=6,
                    help="确认阶段空间覆盖格数门槛（>=6 格 ≈ 覆盖率 0.56）")
    ap.add_argument("--temporal-span", type=int, default=2,
                    help="多帧一致性：对角邻域半径")
    ap.add_argument("--temporal-need", type=int, default=2,
                    help="多帧一致性：通过对角邻域数门槛（0=关闭）")
    ap.add_argument("--temporal-inl", type=int, default=30)
    ap.add_argument("--src-fps", type=float, default=60.0,
                    help="gray 的 idx 是源视频帧号，源 60fps")
    ap.add_argument("--allow-future", action="store_true",
                    help="离线全片口径（允许匹配未来帧），仅作对照")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--dump-npz", type=Path, default=None,
                    help="落盘全量 (inliers, cells) 矩阵，供 loop_gate_sweep 复算门槛")
    args = ap.parse_args()

    import cv2

    z = np.load(args.orb_npz, allow_pickle=True)
    kps = list(z["kps"])
    descs = list(z["descs"])
    BOW = np.load(args.bow).astype(np.float32)
    N = len(descs)
    g = np.load(args.gray)
    gray, idx = g["frames"], g["idx"].astype(float)
    times = idx / args.src_fps
    H, W = gray.shape[1], gray.shape[2]
    wh = (float(W), float(H))
    print(f"[info] frames={N} bow={BOW.shape} {W}x{H} t=[{times[0]:.2f},{times[-1]:.2f}]s")

    # ---- 静止帧检测（必须在候选生成之前）----
    # 判据沿用 overlap_new.py:105-123（已验证实现）：相隔 4 帧的 ORB 匹配点
    # 中位像素位移 > --move-px 才算运动帧；好匹配 < 20 也算不运动。
    # 理由：静止时任意两帧高度相似，未过滤时「98.4% 候选对通过验证」是假象。
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING)
    lag = args.move_lag
    shift = np.full(N, np.nan, dtype=np.float32)
    for i in range(N - lag):
        d1_, d2_ = descs[i], descs[i + lag]
        if d1_ is None or d2_ is None:
            continue
        raw = bfm.knnMatch(d1_, d2_, k=2)
        good = [m[0] for m in raw
                if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < args.move_min_matches:
            continue
        p1_ = np.asarray([kps[i][m.queryIdx] for m in good], np.float32)
        p2_ = np.asarray([kps[i + lag][m.trainIdx] for m in good], np.float32)
        shift[i] = float(np.median(np.linalg.norm(p2_ - p1_, axis=1)))
    moving = np.zeros(N, dtype=bool)
    moving[:N - lag] = np.nan_to_num(shift[:N - lag], nan=0.0) > args.move_px
    static = ~moving
    print(f"[info] 运动帧={int(moving.sum())}/{N}  静止帧={int(static.sum())}  "
          f"(lag={lag} >{args.move_px}px, 有效测距 {int(np.isfinite(shift).sum())})")

    label = np.array([gt_label(t) for t in times], dtype=object)
    menu = np.array([l is None for l in label])
    usable = (~static) & (~menu)
    print(f"[info] 剔除 静止={int(static.sum())} 菜单={int(menu.sum())} "
          f"→ 可用={int(usable.sum())}")

    # ---- 在线正样本定义：存在更早的、已完整结束的同标签段 ----
    def online_earlier(i):
        lab = label[i]
        if lab is None:
            return []
        segs = gt_segments_of(lab)
        return [(a, b) for a, b in segs if b <= times[i] - args.min_sep]

    def hit_of(i, j):
        segs = online_earlier(i)
        if not segs:
            return None
        return any(a - TOL <= times[j] <= b + TOL for a, b in segs)

    queries = [i for i in range(0, N, args.query_stride) if usable[i]]
    n_pos = sum(1 for i in queries if online_earlier(i))
    print(f"[info] queries={len(queries)}  正样本(在线)={n_pos}  "
          f"min_sep={args.min_sep}s  模式={'离线全片' if args.allow_future else '在线历史'}")

    # ---- 候选掩码 ----
    # 在线：只允许过去且已完整结束的段（b <= t_i - min_sep）
    # 离线对照：允许未来，但**两侧**都要排除时间邻域，否则近乎重复的邻帧会
    #           以压倒性内点数占据 top-1，把检索评测彻底污染（实测 oracle@1=0.0）。
    cand_mask = {}
    for i in queries:
        m = usable.copy()
        if args.allow_future:
            m &= np.abs(times - times[i]) >= args.min_sep
        else:
            m &= times <= times[i] - args.min_sep
        m[i] = False
        cand_mask[i] = np.where(m)[0]

    # ---- 全量 ORB 几何基线（同帧集 oracle）----
    t0 = time.time()
    oracle, oracle_lat = {}, []
    INL = np.zeros((len(queries), N), dtype=np.int16)
    CEL = np.zeros((len(queries), N), dtype=np.int8)
    for n, i in enumerate(queries):
        cand = cand_mask[i]
        inl = np.zeros(len(cand), dtype=np.int32)
        tq = time.time()
        for c, j in enumerate(cand):
            v, cl, _ = bidir_verify(kps[i], descs[i], kps[j], descs[j],
                                    args.ratio, args.ransac_px, wh)
            inl[c] = v
            INL[n, j] = v
            CEL[n, j] = cl
        oracle[i] = inl
        oracle_lat.append((time.time() - tq) * 1000.0)
        if n % 10 == 0:
            print(f"[info] oracle {n}/{len(queries)}  {time.time()-t0:.0f}s",
                  file=sys.stderr, flush=True)
    oracle_secs = time.time() - t0
    n_oracle_verif = int(sum(len(cand_mask[i]) for i in queries))
    oracle_lat = np.array(oracle_lat)
    print(f"[info] oracle 全量几何验证 {n_oracle_verif} 次 / {oracle_secs:.0f}s")
    if args.dump_npz:
        np.savez_compressed(args.dump_npz,
                            query_idx=np.array(queries), times=times,
                            label=label.astype(str), usable=usable,
                            moving=moving, inliers=INL, cells=CEL)
        print(f"[saved] {args.dump_npz}  inliers/cells ({len(queries)}x{N})")

    # ---- BoW 检索 ----
    Bn = BOW / (np.linalg.norm(BOW, axis=1, keepdims=True) + 1e-12)
    bow_rank, bow_lat = {}, []
    for i in queries:
        cand = cand_mask[i]
        tq = time.time()
        sim = Bn[cand] @ Bn[i]
        bow_rank[i] = cand[np.argsort(-sim)]
        bow_lat.append((time.time() - tq) * 1000.0)
    bow_lat = np.array(bow_lat)

    # ---- 检索召回 ----
    def recall_at(rank_getter, K):
        ok = tot = 0
        for i in queries:
            if not online_earlier(i):
                continue
            tot += 1
            if any(hit_of(i, int(j)) for j in rank_getter(i)[:K]):
                ok += 1
        return round(ok / tot, 4) if tot else None

    def oracle_rank(i):
        cand = cand_mask[i]
        return cand[np.argsort(-oracle[i])]

    res = {
        "probe_kind": "same-sequence fitted retrieval probe (NOT cross-world)",
        "mode": "offline_full_sequence" if args.allow_future else "online_causal",
        "n_frames": N, "n_queries": len(queries), "n_pos_online": n_pos,
        "static_frames": int(static.sum()), "menu_frames": int(menu.sum()),
        "moving_frames": int(moving.sum()),
        "usable_frames": int(usable.sum()),
        "static_gate": {"lag": args.move_lag, "move_px": args.move_px,
                        "min_matches": args.move_min_matches},
        "min_sep": args.min_sep, "ratio": args.ratio, "ransac_px": args.ransac_px,
        "oracle_verifications": n_oracle_verif,
        "oracle_total_seconds": round(oracle_secs, 1),
        "oracle_ms_p50": round(float(np.percentile(oracle_lat, 50)), 1),
        "oracle_ms_p95": round(float(np.percentile(oracle_lat, 95)), 1),
        "bow_retrieval_ms_p50": round(float(np.percentile(bow_lat, 50)), 2),
        "bow_retrieval_ms_p95": round(float(np.percentile(bow_lat, 95)), 2),
        "avg_candidates_per_query": round(n_oracle_verif / max(len(queries), 1), 1),
    }

    res["retrieval"] = {}
    for K in (1, 5, 10, 20):
        res["retrieval"][f"orb_oracle@{K}"] = recall_at(oracle_rank, K)
        res["retrieval"][f"bow@{K}"] = recall_at(lambda i: bow_rank[i], K)
    res["acceptance_targets_95pct"] = {
        f"bow@{K} >= 0.95 * orb_oracle@{K}": round(
            0.95 * (res["retrieval"][f"orb_oracle@{K}"] or 0), 4)
        for K in (10, 20)}

    # ---- BoW Top-K → 几何 + 覆盖 + 多帧一致性 → 三类结果 ----
    def temporal_support(i, j, span, need, t_inl):
        ok = tried = 0
        for k in range(-span, span + 1):
            if k == 0:
                continue
            a, b = i + k, j + k
            if a < 0 or b < 0 or a >= N or b >= N:
                continue
            if not (usable[a] and usable[b]):
                continue
            tried += 1
            inl, _, _ = bidir_verify(kps[a], descs[a], kps[b], descs[b],
                                     args.ratio, args.ransac_px, wh)
            if inl >= t_inl:
                ok += 1
        return (ok >= need) if need > 0 else True, ok, tried

    confirm = {}
    for K in (5, 10, 20):
        acc_ok = acc_wrong = rej = 0
        n_verif = n_temporal_calls = 0
        lat = []
        per_label: dict = {}
        for i in queries:
            if not online_earlier(i):
                continue
            lab = label[i]
            tq = time.time()
            best, best_inl = None, -1
            for j in bow_rank[i][:K]:
                j = int(j)
                inl, cells, _ = bidir_verify(kps[i], descs[i], kps[j], descs[j],
                                             args.ratio, args.ransac_px, wh)
                n_verif += 1
                if inl >= args.t_inl and cells >= args.cells and inl > best_inl:
                    best, best_inl = j, inl
            # 多帧一致性只在几何+覆盖已通过时才有意义（省算力）
            if best is not None and args.temporal_need > 0:
                n_temporal_calls += 1
                ok_t, _, _ = temporal_support(i, best, args.temporal_span,
                                              args.temporal_need, args.temporal_inl)
                if not ok_t:
                    best = None
            lat.append((time.time() - tq) * 1000.0)
            d = per_label.setdefault(lab, {"n": 0, "ok": 0, "wrong": 0, "rej": 0})
            d["n"] += 1
            if best is None:
                rej += 1
                d["rej"] += 1
            elif hit_of(i, best):
                acc_ok += 1
                d["ok"] += 1
            else:
                acc_wrong += 1
                d["wrong"] += 1
        lat = np.array(lat)
        tot = acc_ok + acc_wrong + rej
        confirm[f"top{K}"] = {
            "n_pos_queries": tot,
            "accepted_correct": acc_ok, "accepted_wrong": acc_wrong,
            "rejected": rej,
            "accept_recall": round(acc_ok / tot, 4) if tot else None,
            "wrong_accept_rate": round(acc_wrong / tot, 4) if tot else None,
            "geometric_verifications": n_verif,
            "avg_verifications_per_query": round(n_verif / tot, 2) if tot else None,
            "temporal_rechecks": n_temporal_calls,
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 1) if len(lat) else None,
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if len(lat) else None,
            "per_label": {k: {kk: (round(vv / v["n"], 4) if kk != "n" else vv)
                              for kk, vv in v.items()}
                          for k, v in sorted(per_label.items())},
        }
    res["confirmation"] = confirm
    res["confirmation_gate"] = {"t_inl": args.t_inl, "cells": args.cells,
                                "temporal_span": args.temporal_span,
                                "temporal_need": args.temporal_need,
                                "temporal_inl": args.temporal_inl}

    print(json.dumps({k: v for k, v in res.items() if k != "confirmation"},
                     ensure_ascii=False, indent=2))
    print(json.dumps({"confirmation": res["confirmation"]},
                     ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"[saved] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
