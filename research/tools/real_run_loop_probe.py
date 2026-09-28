"""GT-free 真实录屏回环探针（单条 run 的参数化前端）。

本脚本是 `research/tools/episode_segment.py` / `research/tools/loop_frame_index.py` /
`research/tools/episode_loop_eval.py` / `research/tools/bow_loop_eval.py` 的 *免人工标注* 替代输入：
那些原工具都依赖 `research/tools/seqslam_probe.py` 里逐帧人工标注的真实回环 ground truth
（其导出符号提供标签与分段查询），因此真实 VRChat 录屏没有这些标注，无法直接喂入
原流水线。

本工具**不引入也不引用**任何人工标注查询符号，完全基于几何与外观：
  1. 按 --stride 抽帧（默认 60fps -> 20fps），缩放为 640x360 灰度；
  2. 每帧提 ORB（nfeatures=600）；
  3. 用本 run 自己的描述子做 k-means 词袋（BoW）词表（VOCAB=400 词）；
  4. 每帧算 BoW 直方图（L1 归一化为频率），检索时按余弦相似度取 Top-K，
     且候选帧对的时隙 >= MIN_GAP；
  5. 对每个候选帧对做几何验证：先用 ratio test (0.8) 做 Lowe 双向 ORB 匹配，
     再用 cv2.findEssentialMat 的 RANSAC（阈值 1.0px，focal=320，主点=(320,180)）
     数内点；内点数 >= MIN_INLIERS 即判为「已验证回环」。
     **注意**：此处只做本质矩阵 RANSAC 内点计数，并不做 cheirality / recoverPose
     位姿恢复，也不做极线约束的二次筛选。该内点计数指标在近似静止 / 单地点的素材上
     区分度很差——均匀随机的非相邻帧对（|i-j|>=60）也有约 25% 能通过内点门槛，
     而一个合格的回环验证器应接近 0–2%。因此 n_verified 只反映噪声地板，
     单凭计数不能作为真实回环存在的证据，须结合运动幅度 / 地点多样性判读。

常量与几何验证逻辑严格复用 `.slam_probe/offline_probe/overlap_probe.py`，
以便与原离线探针保持可比。

用法：
  python research/tools/real_run_loop_probe.py --video VIDEO.mkv --out-dir OUT [--contact-sheet]
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

# 与原 overlap_probe.py 完全一致的相机内参（参见 .slam_probe/REPORT.md）
FX = FY = 320.0
CX, CY = 320.0, 180.0


def parse_args():
    ap = argparse.ArgumentParser(
        description="GT-free real-run loop probe (single run).")
    ap.add_argument("--video", required=True, help="path to .mkv recording")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--stride", type=int, default=3, help="sample every Nth frame")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--orb-n", type=int, default=600, help="ORB nfeatures")
    ap.add_argument("--vocab", type=int, default=400, help="BoW vocabulary size")
    ap.add_argument("--topk", type=int, default=10, help="retrieval neighbors")
    ap.add_argument("--min-gap", type=int, default=60,
                    help="min sampled-frame gap for a candidate pair")
    ap.add_argument("--min-inliers", type=int, default=30,
                    help="min essential-matrix RANSAC inliers to verify")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="optional cap on number of sampled frames")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="write contact_sheet.png of top verified pairs")
    return ap.parse_args()


def load_frames(video, stride, w, h, max_frames):
    """Sample every `stride`-th frame as 640x360 grayscale.

    Returns (gray_frames, src_idx, src_frame_count, fps) where src_idx[k] is the
    source frame index of sampled frame k. src_frame_count is the total number
    of source frames decoded from the video.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError("cannot open video: %s" % video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 60.0  # sensible fallback; recordings are 60fps
    frames, src_idx = [], []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if max_frames is not None and len(frames) >= max_frames:
            break
        if i % stride == 0:
            small = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            src_idx.append(i)
        i += 1
    cap.release()
    src_frame_count = i
    return frames, src_idx, src_frame_count, fps


def extract_features(frames, orb_n):
    """ORB detectAndCompute on every gray frame (params mirror overlap_probe)."""
    orb = cv2.ORB_create(
        nfeatures=orb_n, scaleFactor=1.2, nlevels=8,
        edgeThreshold=31, firstLevel=0, WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=12,
    )
    kps, descs = [], []
    for g in frames:
        k, d = orb.detectAndCompute(g, None)
        if k is None or len(k) == 0:
            kps.append(np.zeros((0, 2), np.float32))
            descs.append(None)
        else:
            kps.append(np.float32([kp.pt for kp in k]))
            descs.append(d)
    return kps, descs


def build_vocab(descs, vocab_size):
    """k-means BoW vocabulary built from this run's own descriptors."""
    rng = np.random.default_rng(0)
    sample = []
    for d in descs:
        if d is None or len(d) == 0:
            continue
        take = min(80, len(d))
        sample.append(d[rng.choice(len(d), take, replace=False)])
    if not sample:
        raise RuntimeError("no descriptors available to build vocabulary")
    sample = np.vstack(sample).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(
        sample, vocab_size, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    return centers.astype(np.uint8)


def build_bow(descs, vocab):
    """Per-frame L1-normalized BoW histogram (tf-idf weighted)."""
    n = len(descs)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    bow = np.zeros((n, vocab.shape[0]), np.float32)
    for i, d in enumerate(descs):
        if d is None or len(d) == 0:
            continue
        m = matcher.match(d.astype(np.uint8), vocab)
        for mm in m:
            bow[i, mm.trainIdx] += 1.0
    # tf-idf weighting (common words down-weighted)
    df = (bow > 0).sum(axis=0).astype(np.float64)
    idf = np.log((n + 1.0) / (df + 1.0)).astype(np.float32)
    bow = bow * idf
    # L1 normalize -> frequency histogram (sums to 1 per frame)
    rowsum = bow.sum(axis=1, keepdims=True) + 1e-9
    bow = bow / rowsum
    return bow


def cosine_sim(bow):
    """Cosine similarity matrix (true cosine on the L1-normalized histogram)."""
    norms = np.linalg.norm(bow, axis=1, keepdims=True) + 1e-9
    n = bow / norms
    return n @ n.T


def verify(i, j, kps, descs, min_inliers):
    """Essential-matrix RANSAC verification. Returns (n_matches, n_inliers).

    EXACT logic from overlap_probe.py. A pair is verified when n_inliers >= min_inliers.
    """
    d1, d2 = descs[i], descs[j]
    if d1 is None or d2 is None or len(d1) < 50 or len(d2) < 50:
        return 0, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(d1, d2, k=2)
    good = [x[0] for x in raw if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
    if len(good) < min_inliers:
        return len(good), 0
    p1 = np.asarray(kps[i], np.float32)[[g.queryIdx for g in good]]
    p2 = np.asarray(kps[j], np.float32)[[g.trainIdx for g in good]]
    K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
    E, mask = cv2.findEssentialMat(
        p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None or mask is None:
        return len(good), 0
    return len(good), int(mask.ravel().astype(bool).sum())


def write_contact_sheet(out_path, pairs, gray_frames, src_idx, w, h,
                        max_rows=12):
    """Compose contact sheet with cv2 only (matplotlib not available)."""
    rows = []
    for pr in pairs[:max_rows]:
        i, j = pr["i"], pr["j"]
        fi = gray_frames[i] if i < len(gray_frames) else None
        fj = gray_frames[j] if j < len(gray_frames) else None
        if fi is None or fj is None:
            continue
        # upscale a touch for legibility
        sc = 1.0
        fi = cv2.resize(fi, (int(w * sc), int(h * sc)))
        fj = cv2.resize(fj, (int(w * sc), int(h * sc)))
        pair_img = np.hstack([fi, fj])
        # header bar with annotation
        bar = 30
        header = np.full((bar, pair_img.shape[1]), 20, np.uint8)
        label = "i=%d j=%d gap_s=%.2f inliers=%d" % (
            i, j, pr["gap_s"], pr["inliers"])
        cv2.putText(header, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, 255, 1, cv2.LINE_AA)
        # divider between the two frames
        mid = fi.shape[1]
        pair_img[:, mid - 1:mid + 1] = 160
        rows.append(np.vstack([header, pair_img]))
    if not rows:
        # placeholder if nothing to show
        ph = np.zeros((60, 400), np.uint8)
        cv2.putText(ph, "no verified pairs", (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 200, 1, cv2.LINE_AA)
        rows.append(ph)
    sheet = np.vstack(rows)
    cv2.imwrite(out_path, sheet)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- default params echoed in output ---
    params = {
        "stride": args.stride,
        "width": args.width,
        "height": args.height,
        "orb_n": args.orb_n,
        "vocab": args.vocab,
        "topk": args.topk,
        "min_gap": args.min_gap,
        "min_inliers": args.min_inliers,
        "fx": FX, "fy": FY, "cx": CX, "cy": CY,
    }

    # --- empty-result helper so we always write a valid JSON ---
    empty = {
        "video": args.video,
        "fps": None,
        "n_frames": 0,
        "src_frame_count": 0,
        "stride": args.stride,
        "params": params,
        "n_verified": 0,
        "pairs": [],
        "note": "",
    }

    # 1) load frames
    try:
        frames, src_idx, src_frame_count, fps = load_frames(
            args.video, args.stride, args.width, args.height, args.max_frames)
    except Exception as e:
        empty["note"] = "video open/read failed: %s" % e
        out = os.path.join(args.out_dir, "loop_probe.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(empty, f, indent=1)
        print("ERROR: %s" % empty["note"])
        print("wrote (empty) %s" % out)
        if args.contact_sheet:
            write_contact_sheet(os.path.join(args.out_dir, "contact_sheet.png"),
                                [], frames, src_idx, args.width, args.height)
        return

    n = len(frames)
    print("frames: %d sampled (stride %d)  src_frame_count=%d  fps=%.2f"
          % (n, args.stride, src_frame_count, fps))

    # 2) features
    kps, descs = extract_features(frames, args.orb_n)
    nfeat = [0 if d is None else len(d) for d in descs]
    print("features: mean %.0f  min %d  max %d"
          % (np.mean(nfeat), np.min(nfeat), np.max(nfeat)))

    if n == 0 or max(nfeat) == 0:
        empty["fps"] = float(fps)
        empty["n_frames"] = 0
        empty["src_frame_count"] = src_frame_count
        empty["note"] = "no frames / no ORB features extracted"
        out = os.path.join(args.out_dir, "loop_probe.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(empty, f, indent=1)
        print("ERROR: %s" % empty["note"])
        print("wrote (empty) %s" % out)
        if args.contact_sheet:
            write_contact_sheet(os.path.join(args.out_dir, "contact_sheet.png"),
                                [], frames, src_idx, args.width, args.height)
        return

    # 3) vocabulary + BoW + cosine retrieval
    vocab = build_vocab(descs, args.vocab)
    bow = build_bow(descs, vocab)
    sim = cosine_sim(bow)

    pairs = []
    t0 = time.time()
    for i in range(n):
        order = np.argsort(-sim[i])
        picked = 0
        for j in order:
            j = int(j)
            if abs(j - i) < args.min_gap:
                continue
            pairs.append((i, j))
            picked += 1
            if picked >= args.topk:
                break
    print("candidate pairs: %d  (%.1fs)" % (len(pairs), time.time() - t0))

    # 4) essential-matrix verification
    results = []
    t0 = time.time()
    for k, (i, j) in enumerate(pairs):
        nmatch, ninl = verify(i, j, kps, descs, args.min_inliers)
        if ninl >= args.min_inliers:
            t_i = i * args.stride / fps
            t_j = j * args.stride / fps
            results.append({
                "i": int(i), "j": int(j),
                "t_i": float(t_i), "t_j": float(t_j),
                "gap_s": float(abs(t_i - t_j)),
                "inliers": int(ninl),
            })
        if k % 2000 == 0 and k:
            print("  verify %d/%d  %.1fs" % (k, len(pairs), time.time() - t0))

    results.sort(key=lambda r: -r["inliers"])
    print("verified pairs: %d / %d candidates  (%.1fs)"
          % (len(results), len(pairs), time.time() - t0))

    out = {
        "video": args.video,
        "fps": float(fps),
        "n_frames": n,
        "src_frame_count": src_frame_count,
        "stride": args.stride,
        "params": params,
        "n_verified": len(results),
        "pairs": results,
    }
    out_path = os.path.join(args.out_dir, "loop_probe.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote %s" % out_path)

    if args.contact_sheet:
        cs_path = os.path.join(args.out_dir, "contact_sheet.png")
        write_contact_sheet(cs_path, results, frames, src_idx,
                            args.width, args.height)
        print("wrote %s" % cs_path)

    # 5) human-readable summary
    print("\n=== summary ===")
    print("video          : %s" % args.video)
    print("fps            : %.2f" % fps)
    print("sampled frames : %d" % n)
    print("verified loops : %d" % len(results))
    print("top-5 pairs (i, j, gap_s, inliers):")
    for r in results[:5]:
        print("   i=%d j=%d gap_s=%.2f inliers=%d"
              % (r["i"], r["j"], r["gap_s"], r["inliers"]))
    if not results:
        print("   (none)")


if __name__ == "__main__":
    main()
