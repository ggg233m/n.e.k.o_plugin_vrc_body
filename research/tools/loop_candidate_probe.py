"""回环候选探针：算出 ORB 内点矩阵 + 空间覆盖指标 + color 描述子，落盘供级联评测复算。

**只做计算，不做评测**（评测在 loop_cascade_eval.py，改阈值不用重跑匹配）。

用法
----
  .venv/Scripts/python.exe research/tools/loop_candidate_probe.py \
      --frames-dir .tmp/seq_frames --query-stride 3 --ratio 0.75 \
      --out .tmp/loop_candidates.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.tools.orb_place_probe import match_pair, _orb  # noqa: E402
from backend.avatar_identity import appearance_descriptor  # noqa: E402


def _spatial_metrics(pq: np.ndarray, inlier_mask: np.ndarray,
                     grid: int = 3) -> tuple[float, float, float, int]:
    """由查询帧侧的内点坐标算 4 个空间覆盖指标。

    返回 (格覆盖率, 凸包面积比, 最大格占比, 去掉最大格后剩余内点数)。
    """
    pts = pq[inlier_mask]
    n = len(pts)
    if n < 3:
        return 0.0, 0.0, 1.0, 0
    # 格覆盖 / 最大格占比
    cell = np.floor(np.clip(pts, 0.0, 0.999999) * grid).astype(int)
    flat = cell[:, 0] * grid + cell[:, 1]
    counts = np.bincount(flat, minlength=grid * grid)
    occupied = int((counts > 0).sum())
    dominant = int(counts.max())
    coverage = occupied / (grid * grid)
    dom_frac = dominant / n
    remaining = n - dominant
    # 凸包面积比（归一化坐标，面积即占画面比例）
    try:
        import cv2
        hull = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2))
        area = float(cv2.contourArea(hull))
    except Exception:
        area = 0.0
    return float(coverage), float(area), float(dom_frac), int(remaining)


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--query-stride", type=int, default=3)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d: Path = args.frames_dir
    files = sorted(d.glob("*.jpg"))
    times = json.loads((d / "times.json").read_text(encoding="utf-8"))[: len(files)]
    N = len(files)

    orb = _orb()
    feats, colors = [], []
    for p in files:
        im = cv2.imread(str(p))
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (960, 540), interpolation=cv2.INTER_AREA)
        k, de = orb.detectAndCompute(g, None)
        feats.append((k, de))
        v = np.asarray(appearance_descriptor(im, (0.0, 0.0, 1.0, 1.0)),
                       dtype=np.float32).reshape(-1)
        v = v / (np.linalg.norm(v) + 1e-12)
        colors.append(v)
    C = np.stack(colors)

    qs = list(range(0, N, args.query_stride))
    M = len(qs)
    INL = np.zeros((M, N), dtype=np.int32)      # 内点数（越大越像同一地点）
    COV = np.zeros((M, N), dtype=np.float32)    # 格覆盖率
    HULL = np.zeros((M, N), dtype=np.float32)   # 凸包面积比
    DOM = np.ones((M, N), dtype=np.float32)     # 最大格占比（越小越分散）
    REM = np.zeros((M, N), dtype=np.int32)      # 去掉最大格后剩余内点数

    t0 = time.time()
    for a, i in enumerate(qs):
        for j in range(N):
            if j == i:
                continue
            good, inl, _, pq, pc, mask = match_pair(
                feats[i][1], feats[j][1], feats[i][0], feats[j][0],
                args.ratio, return_pts=True)
            INL[a, j] = inl
            if inl >= 3 and pq is not None:
                c, ar, df, rem = _spatial_metrics(pq, mask)
                COV[a, j], HULL[a, j], DOM[a, j], REM[a, j] = c, ar, df, rem
        if a % 20 == 0:
            print(f"[info] {a}/{M}  {time.time() - t0:.0f}s", file=sys.stderr)

    np.savez_compressed(
        args.out, times=np.array(times), query_idx=np.array(qs),
        inliers=INL, coverage=COV, hull=HULL, dominant=DOM, remaining=REM,
        color=C)
    print(f"[saved] {args.out}  ({M}x{N})  {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
