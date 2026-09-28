"""因果多帧投票探针：为确认层算出 (内点, 覆盖, 最大局部簇, 剩余内点) 随**时间滞后**的曲线。

为什么单独写
------------
`bow_loop_eval.py` 里的多帧检查用 ``span=2``，在 20fps 下只有 **0.1 秒**——
那是"同一帧的第二份拷贝"，不是多帧投票。真正的因果多帧投票必须跨 0.5~1.5 秒，
即滞后取 10 / 20 / 30 帧。

因果性约束（用户 2026-09-20 定稿）
--------------------------------
- 投票帧只能是 **当前帧及其过去帧**：``a = i - lag``，``b = j - lag``（lag ≥ 0）
- **不允许未来帧**
- 常滞后对 (i,j) 的因果合法性在平移下保持：若 ``t_j ≤ t_i - min_sep``，
  则 ``t_b = t_j - lag·dt ≤ t_i - lag·dt - min_sep = t_a - min_sep`` 仍成立，
  所以投票对天然满足"候选来自过去"。**这不是额外规则，是恒等式。**

对角匹配 = 常时间偏移一致性
--------------------------
多帧投票与"路线历史一致"在实现上是同一件事：一对 (i,j) 隐含一个行进偏移
``d = j - i``。要求 (i-lag, j-lag) 也成立，就是要求**同一个偏移在时间上稳定**。
本探针把这个量显式化，不引入任何额外的朝向积分（AngularY 是**指令回声**，不是
实际转角测量，无单一定标系数，不能积分出朝向——见 `.workbuddy/memory/2026-09-19.md`）。

输出
----
``.npz``：INL/CEL/DOM/REM 形状 ``(n_pos, topk, n_lag)`` + 候选帧号 + 查询帧号 + 滞后表。

用法
----
  .venv/Scripts/python.exe research/tools/loop_vote_probe.py \
      --query-stride 8 --topk 20 --lags 0,10,20,30 \
      --out .tmp/loop_vote_pairs.npz
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

from research.tools.seqslam_probe import gt_label, gt_segments_of  # noqa: E402

TOL = 1.0
GRID_COV = 3        # 覆盖率用 3x3 → 格数 0..9
GRID_CLU = 4        # 局部簇用 4x4 → 更细，用于"去掉最大簇后剩余"


def verify(kp1, d1, kp2, d2, ratio: float, ransac_px: float,
           wh: tuple[float, float]):
    """双向匹配 + 基础矩阵 RANSAC。

    返回 ``(inl, cells, dom, rem)``：
    - ``cells`` 查询帧侧内点覆盖的 3x3 格数
    - ``dom``   4x4 网格里最大格的占比（越大越集中 = 越像重复纹理）
    - ``rem``   去掉 4x4 最大格后**剩余内点数**（pooldeck 类重复纹理的解药指标）
    """
    import cv2
    if d1 is None or d2 is None or len(d1) < 20 or len(d2) < 20:
        return 0, 0, 1.0, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def one_way(a, b):
        raw = bf.knnMatch(a, b, k=2)
        return {m[0].queryIdx: m[0].trainIdx
                for m in raw if len(m) == 2 and m[0].distance < ratio * m[1].distance}

    fwd = one_way(d1, d2)
    bwd = one_way(d2, d1)
    mutual = [(qi, ti) for qi, ti in fwd.items() if bwd.get(ti) == qi]
    if len(mutual) < 8:
        return 0, 0, 1.0, 0
    p1 = np.asarray([kp1[qi] for qi, _ in mutual], dtype=np.float32)
    p2 = np.asarray([kp2[ti] for _, ti in mutual], dtype=np.float32)
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, ransac_px, 0.99)
    if F is None or mask is None:
        return 0, 0, 1.0, 0
    m = mask.ravel().astype(bool)
    inl = int(m.sum())
    if inl < 3:
        return inl, 0, 1.0, 0
    nq = p1[m] / np.asarray(wh, dtype=np.float32)
    nq = np.clip(nq, 0.0, 0.999999)
    c3 = np.floor(nq * GRID_COV).astype(int)
    cells = int((np.bincount(c3[:, 0] * GRID_COV + c3[:, 1],
                             minlength=GRID_COV ** 2) > 0).sum())
    c4 = np.floor(nq * GRID_CLU).astype(int)
    counts = np.bincount(c4[:, 0] * GRID_CLU + c4[:, 1], minlength=GRID_CLU ** 2)
    biggest = int(counts.max())
    return inl, cells, biggest / inl, inl - biggest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orb-npz", type=Path,
                    default=Path(".slam_probe/offline_probe/orb_full_n600.npz"))
    ap.add_argument("--bow", type=Path,
                    default=Path(".slam_probe/offline_probe/bow_full_v400.npy"))
    ap.add_argument("--gray", type=Path,
                    default=Path(".slam_probe/offline_probe/gray_640x360_s3_full.npz"))
    ap.add_argument("--query-stride", type=int, default=8)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--lags", type=str, default="0,10,20,30",
                    help="投票滞后（帧，20fps 下 10 帧=0.5s）。必须含 0。")
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

    lags = [int(x) for x in args.lags.split(",") if x.strip() != ""]
    if 0 not in lags:
        raise SystemExit("--lags 必须包含 0")
    lags = sorted(lags)

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
    print(f"[info] frames={N} bow={BOW.shape} {W}x{H} lags={lags}")

    # 静止帧门（overlap_new.py 判据，与 bow_loop_eval 一致）
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
    print(f"[info] 运动={int(moving.sum())} 满足=可用 {int(usable.sum())}")

    def earlier(i):
        L = label[i]
        if L is None:
            return []
        return [(a, b) for a, b in gt_segments_of(L) if b <= times[i] - args.min_sep]

    queries = [i for i in range(0, N, args.query_stride) if usable[i]]
    pos = [i for i in queries if earlier(i)]
    print(f"[info] queries={len(queries)}  在线正样本={len(pos)}")

    Bn = BOW / (np.linalg.norm(BOW, axis=1, keepdims=True) + 1e-12)
    L = len(lags)
    INL = np.zeros((len(pos), args.topk, L), dtype=np.int16)
    CEL = np.zeros((len(pos), args.topk, L), dtype=np.int8)
    DOM = np.ones((len(pos), args.topk, L), dtype=np.float32)
    REM = np.zeros((len(pos), args.topk, L), dtype=np.int16)
    CAND = np.full((len(pos), args.topk), -1, dtype=np.int32)
    QIDX = np.array(pos, dtype=np.int32)
    # 投票帧对是否可用（未越界 / 非静止 / 非菜单）
    VOK = np.zeros((len(pos), args.topk, L), dtype=bool)
    # 投票对与查询帧是否属于同一处访问（路线连续性代理；仅用于离线诊断）
    VSAME = np.zeros((len(pos), args.topk, L), dtype=bool)

    t0 = time.time()
    npair = 0
    for r, i in enumerate(pos):
        cand = np.where(usable & (times <= times[i] - args.min_sep))[0]
        cand = cand[cand != i]
        if cand.size == 0:
            continue
        sim = Bn[cand] @ Bn[i]
        top = cand[np.argsort(-sim)][:args.topk]
        CAND[r, :len(top)] = top
        for c, j in enumerate(top):
            for li, lag in enumerate(lags):
                a, b = i - lag, j - lag
                if a < 0 or b < 0:
                    continue
                if not (usable[a] and usable[b]):
                    continue
                VOK[r, c, li] = True
                VSAME[r, c, li] = (label[a] is not None and label[a] == label[i])
                inl, cells, dom, rem = verify(kps[a], descs[a], kps[b], descs[b],
                                              args.ratio, args.ransac_px, wh)
                INL[r, c, li] = inl
                CEL[r, c, li] = cells
                DOM[r, c, li] = dom
                REM[r, c, li] = rem
                npair += 1
        if r % 10 == 0:
            print(f"[info] {r}/{len(pos)}  {time.time()-t0:.0f}s  pairs={npair}",
                  file=sys.stderr, flush=True)

    np.savez_compressed(
        args.out, query_idx=QIDX, cand=CAND, lags=np.array(lags),
        inliers=INL, cells=CEL, dominant=DOM, remaining=REM,
        vote_ok=VOK, vote_same_segment=VSAME,
        times=times, label=label.astype(str), usable=usable,
        min_sep=args.min_sep, ratio=args.ratio, ransac_px=args.ransac_px,
        src_fps=args.src_fps, query_stride=args.query_stride, topk=args.topk)
    print(f"[saved] {args.out}  pairs={npair}  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
