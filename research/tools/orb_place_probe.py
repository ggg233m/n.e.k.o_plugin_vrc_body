"""ORB 局部特征地点识别的离线基线探针（不接实时链路）。

与 ``seqslam_probe.py`` 用同一套素材和真值，回答第二候选路线的问题：
**局部特征（ORB）能不能认出重访，以及低纹理/动漫渲染/镜子会不会让它失效。**

方法
----
每帧提 ORB 关键点与描述子；帧对之间用 BFMatcher + ratio test 得到匹配数，
再用 ``findFundamentalMat``(RANSAC) 做几何验证取内点数。
用**基础矩阵**而不是单应矩阵：回环两段是**反向**行进的，平面单应不成立，
而基础矩阵与朝向无关。

真值复用 ``seqslam_probe.GT_SEGMENTS``：A/B/C/D 各有一次重访（正样本），
E/F 是单访段（任何命中都是误报）。

用法
----
  .venv/Scripts/python.exe research/tools/orb_place_probe.py \
      --frames-dir .tmp/seq_frames --query-stride 3 --ratio 0.75 \
      --json-out .tmp/orb_place_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.tools.seqslam_probe import (  # noqa: E402
    GT_SEGMENTS, gt_label, gt_segments_of,
)


def _orb():
    import cv2
    return cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8,
                          edgeThreshold=15, fastThreshold=8)


def match_pair(d1, d2, k1, k2, ratio: float, return_pts: bool = False):
    """返回 (good 数, 几何内点数, 内点率)；return_pts=True 时改返回
    (good, inliers, ratio, pts_query, pts_cand, inlier_mask)，坐标已按画面尺寸归一化。

    用**基础矩阵**而非单应：回环两段常是反向行进，平面单应不成立。
    """
    import cv2
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return (0, 0, 0.0, None, None, None) if return_pts else (0, 0, 0.0)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        knn = bf.knnMatch(d1, d2, k=2)
    except cv2.error:
        return (0, 0, 0.0, None, None, None) if return_pts else (0, 0, 0.0)
    good = []
    for m in knn:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good.append(m[0])
    if len(good) < 8:
        return ((len(good), 0, 0.0, None, None, None) if return_pts
                else (len(good), 0, 0.0))
    # 归一化坐标（相对 960x540 的 ORB 输入尺寸）
    p1 = np.float32([k1[m.queryIdx].pt for m in good]) / np.float32([960.0, 540.0])
    p2 = np.float32([k2[m.trainIdx].pt for m in good]) / np.float32([960.0, 540.0])
    try:
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 3.0 / 960.0, 0.99)
    except cv2.error:
        return ((len(good), 0, 0.0, None, None, None) if return_pts
                else (len(good), 0, 0.0))
    if F is None or mask is None:
        return ((len(good), 0, 0.0, None, None, None) if return_pts
                else (len(good), 0, 0.0))
    m = mask.ravel().astype(bool)
    inl = int(m.sum())
    if return_pts:
        return len(good), inl, (inl / len(good)) if good else 0.0, p1, p2, m
    return len(good), inl, (inl / len(good)) if good else 0.0


def cmd_run(args: argparse.Namespace) -> int:
    import cv2

    d: Path = args.frames_dir
    files = sorted(d.glob("*.jpg"))
    if not files:
        print(f"[error] 没有帧: {d}", file=sys.stderr)
        return 2
    tp = d / "times.json"
    times = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else \
        [i / args.fps for i in range(len(files))]
    times = times[:len(files)]

    orb = _orb()
    feats = []
    kp_counts = []
    for p in files:
        im = cv2.imread(str(p))
        if im is None:
            feats.append((None, None))
            kp_counts.append(0)
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (960, 540), interpolation=cv2.INTER_AREA)
        k, de = orb.detectAndCompute(g, None)
        feats.append((k, de))
        kp_counts.append(0 if k is None else len(k))
    kc = np.array(kp_counts)
    print(f"[info] frames={len(feats)} 关键点 min={kc.min()} "
          f"median={int(np.median(kc))} max={kc.max()} "
          f"低于200的帧={(kc < 200).sum()}")

    N = len(feats)
    # 分数矩阵：内点数越多越像同一地点 → 距离 = -inliers
    S = np.full((N, N), np.inf, dtype=np.float32)
    for i in range(0, N, args.query_stride):
        for j in range(N):
            if j == i:
                continue
            _, inl, _ = match_pair(feats[i][1], feats[j][1],
                                   feats[i][0], feats[j][0], args.ratio)
            S[i, j] = -float(inl)
        if i % (args.query_stride * 10) == 0:
            print(f"[info] query {i}/{N}", file=sys.stderr)

    # 评测（与 SeqSLAM 同一套口径）
    rows = []
    for i in range(0, N, args.query_stride):
        ti = times[i]
        li = gt_label(ti)
        if li is None:
            continue
        valid = [j for j in range(N) if j != i
                 and abs(times[j] - ti) >= args.min_sep
                 and np.isfinite(S[i, j])]
        if not valid:
            continue
        sc = np.array([S[i, j] for j in valid])
        order = np.argsort(sc, kind="stable")
        cand_t = [times[valid[int(x)]] for x in order[: args.topk]]
        best_t = cand_t[0]
        segs = gt_segments_of(li)
        rev = len(segs) >= 2

        def _hit(t: float) -> bool:
            for a, b in segs:
                if a - 1.0 <= ti <= b + 1.0:
                    continue
                if a - 1.0 <= t <= b + 1.0:
                    return True
            return False

        hit = rev and _hit(best_t)
        hits_k = [rev and any(_hit(t) for t in cand_t[:k])
                  for k in (1, 5, 10, 20)]
        rows.append({"t": ti, "label": li, "revisited": rev,
                     "best_t": best_t, "cand_top20": cand_t[:20],
                     "score": float(sc[order[0]]), "hit_top1": hit,
                     "hit_topk": hits_k})

    pos = [r for r in rows if r["revisited"]]
    neg = [r for r in rows if not r["revisited"]]
    rep: dict = {
        "n_frames": N, "query_stride": args.query_stride, "ratio": args.ratio,
        "keypoints": {"min": int(kc.min()), "median": int(np.median(kc)),
                      "max": int(kc.max()),
                      "frames_below_200": int((kc < 200).sum())},
        "n_pos": len(pos), "n_neg": len(neg),
        "recall_top1": (sum(r["hit_top1"] for r in pos) / len(pos)) if pos else None,
        "recall_topk": {
            f"@{k}": (sum(r["hit_topk"][i] for r in pos) / len(pos)) if pos else None
            for i, k in enumerate((1, 5, 10, 20))},
    }
    by: dict[str, list[bool]] = {}
    for r in pos:
        by.setdefault(r["label"], []).append(r["hit_top1"])
    byk: dict[str, dict[str, list[bool]]] = {}
    for r in pos:
        d = byk.setdefault(r["label"], {"@1": [], "@5": [], "@10": [], "@20": []})
        for i, k in enumerate(("@1", "@5", "@10", "@20")):
            d[k].append(r["hit_topk"][i])
    rep["recall_topk_by_label"] = {
        k: {kk: round(sum(vv) / len(vv), 4) for kk, vv in v.items()}
        for k, v in sorted(byk.items())}
    if pos and neg:
        sp = np.array([r["score"] for r in pos])
        sn = np.array([r["score"] for r in neg])
        lo, hi = float(min(sp.min(), sn.min())), float(max(sp.max(), sn.max()))
        grid = np.linspace(lo, hi, 200)
        tpr = np.array([(sp <= t).mean() for t in grid])
        fpr = np.array([(sn <= t).mean() for t in grid])
        bal = (tpr + (1 - fpr)) / 2
        bi = int(np.argmax(bal))
        o = np.argsort(fpr)
        rep["pr"] = {
            "best_balanced_acc": round(float(bal[bi]), 4),
            "best_threshold_inliers": round(-float(grid[bi]), 1),
            "tpr_at_best": round(float(tpr[bi]), 4),
            "fpr_at_best": round(float(fpr[bi]), 4),
            "auc": round(float(np.trapezoid(tpr[o], fpr[o])), 4),
        }
        rep["inliers_pos"] = {"min": int(-sp.max()), "median": int(-np.median(sp)),
                              "max": int(-sp.min())}
        rep["inliers_neg"] = {"min": int(-sn.max()), "median": int(-np.median(sn)),
                              "max": int(-sn.min())}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"[saved] {args.json_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--query-stride", type=int, default=3)
    ap.add_argument("--min-sep", type=float, default=8.0)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--json-out", type=Path, default=None)
    return cmd_run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
