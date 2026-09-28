"""全帧 BoW 候选索引：为**偏移无关**的地点级多帧投票准备数据。

为什么需要它
------------
`loop_vote_probe.py` 做的是**对角**多帧投票 ``(i−lag, j−lag)``——它隐含假设
两次访问**同向、同速、同序**走过同一地点。本素材不满足：重访是**反向/变序行进**
（§16 已记录 SeqSLAM 在素材上失效的理由正是"回环全是反向行进"）。
实测：对角投票的配对级富集度有 8.4 倍（真值 6.4% vs 非真值 0.76%），
但查询级收益极小——正是假设不成立的表现。

用户协议要求的是**累计在候选地点上**的连续支持长度，与偏移无关：
同一段查询帧里，每一帧各自独立地去匹配**同一个候选地点**，
连续多少帧能匹配上，就是支持长度。

本脚本为每个可用帧算出它自己的 BoW Top-20 及其几何证据，
之后任何滞后 / 任何查询帧的投票都能直接查表，不用重跑匹配。

用法
----
  .venv/Scripts/python.exe research/tools/loop_frame_index.py \
      --topk 20 --max-frames 1200 --out .tmp/loop_frame_index.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.tools.loop_vote_probe import verify  # noqa: E402
from research.tools.seqslam_probe import gt_label  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orb-npz", type=Path,
                    default=Path(".slam_probe/offline_probe/orb_full_n600.npz"))
    ap.add_argument("--bow", type=Path,
                    default=Path(".slam_probe/offline_probe/bow_full_v400.npy"))
    ap.add_argument("--gray", type=Path,
                    default=Path(".slam_probe/offline_probe/gray_640x360_s3_full.npz"))
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--agg-stride", type=int, default=1,
                    help="建立索引的抽帧步长（1=每一帧都建，最灵活）")
    ap.add_argument("--min-sep", type=float, default=8.0)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac-px", type=float, default=2.0)
    ap.add_argument("--move-lag", type=int, default=4)
    ap.add_argument("--move-px", type=float, default=1.5)
    ap.add_argument("--move-min-matches", type=int, default=20)
    ap.add_argument("--src-fps", type=float, default=60.0)
    ap.add_argument("--out", type=Path, required=True)
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

    # 静止帧门（overlap_new.py 判据）
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING)
    lg = args.move_lag
    shift = np.full(N, np.nan, dtype=np.float32)
    for i in range(N - lg):
        d1_, d2_ = descs[i], descs[i + lg]
        if d1_ is None or d2_ is None:
            continue
        raw = bfm.knnMatch(d1_, d2_, k=2)
        good = [m[0] for m in raw
                if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < args.move_min_matches:
            continue
        p1_ = np.asarray([kps[i][m.queryIdx] for m in good], np.float32)
        p2_ = np.asarray([kps[i + lg][m.trainIdx] for m in good], np.float32)
        shift[i] = float(np.median(np.linalg.norm(p2_ - p1_, axis=1)))
    moving = np.zeros(N, dtype=bool)
    moving[:N - lg] = np.nan_to_num(shift[:N - lg], nan=0.0) > args.move_px
    label = np.array([gt_label(t) for t in times], dtype=object)
    menu = np.array([l is None for l in label])
    usable = moving & (~menu)

    frames = [i for i in range(0, N, args.agg_stride) if usable[i]]
    print(f"[info] frames={N} 可用={int(usable.sum())} 建索引={len(frames)} topk={args.topk}")

    Bn = BOW / (np.linalg.norm(BOW, axis=1, keepdims=True) + 1e-12)
    K = args.topk
    CAND = np.full((len(frames), K), -1, dtype=np.int32)
    INL = np.zeros((len(frames), K), dtype=np.int16)
    CEL = np.zeros((len(frames), K), dtype=np.int8)
    DOM = np.ones((len(frames), K), dtype=np.float32)
    REM = np.zeros((len(frames), K), dtype=np.int16)

    t0 = time.time()
    npair = 0
    for r, a in enumerate(frames):
        cand = np.where(usable & (times <= times[a] - args.min_sep))[0]
        cand = cand[cand != a]
        if cand.size == 0:
            continue
        sim = Bn[cand] @ Bn[a]
        top = cand[np.argsort(-sim)][:K]
        CAND[r, :len(top)] = top
        for c, j in enumerate(top):
            inl, cells, dom, rem = verify(kps[a], descs[a], kps[j], descs[j],
                                          args.ratio, args.ransac_px, wh)
            INL[r, c], CEL[r, c], DOM[r, c], REM[r, c] = inl, cells, dom, rem
            npair += 1
        if r % 100 == 0:
            print(f"[info] {r}/{len(frames)}  {time.time()-t0:.0f}s  pairs={npair}",
                  file=sys.stderr, flush=True)

    # 帧号 → 行号 的查找表（-1 表示未建索引）
    row_of = np.full(N, -1, dtype=np.int32)
    for r, a in enumerate(frames):
        row_of[a] = r

    np.savez_compressed(
        args.out, frames=np.array(frames, dtype=np.int32), row_of=row_of,
        cand=CAND, inliers=INL, cells=CEL, dominant=DOM, remaining=REM,
        times=times, label=label.astype(str), usable=usable,
        min_sep=args.min_sep, ratio=args.ratio, ransac_px=args.ransac_px,
        src_fps=args.src_fps, topk=K, agg_stride=args.agg_stride)
    print(f"[saved] {args.out}  frames={len(frames)} pairs={npair}  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
