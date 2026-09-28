"""地点可区分性离线测量。

本工具回答一个关键问题：在玩家确实走过多个不同视觉区域的前提下，
`_context_descriptor`（4x8 网格背景指纹）和 `appearance_descriptor`（颜色直方图）
能不能把"同一地点"和"不同地点"分开？

它复用 backend.avatar_identity 的现有资产，不修改实时链路。
对 ``2026-09-18 07-31-11.mkv`` 的 32 张关键帧（每 2s 一张），
人工按肉眼观察把每张帧归属到一个粗略区域，然后计算：

- 区域内（intra）相似度：同一区域、不同视角/时刻的帧两两相似度
- 区域间（inter）相似度：不同区域帧两两相似度
- 候选阈值的可分离性：两者是否重叠

用法
----
  # 默认使用仓库根下的 tmp/keyframes（32 张 2s 关键帧）
  python research/tools/location_distinguishability_probe.py

  # 指定帧目录
  python research/tools/location_distinguishability_probe.py --frames-dir ./my_frames
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Sequence


# --------------------------------------------------------------------------- #
# 人工区域标注（基于 2026-09-18 07-31-11.mkv 每 2s 一张共 32 张关键帧的肉眼观察）
# --------------------------------------------------------------------------- #
# 帧索引 1..32 对应时间 0s,2s,...,62s。
# 这些标签是人在看过画面后给出的粗略区域归属；脚本会如实报告此先验。
DEFAULT_LABELS: dict[int, str] = {
    1: "main_hall",
    2: "main_hall",
    3: "main_hall",
    4: "portal_area",
    5: "portal_area",
    6: "portal_area",
    7: "portal_area",
    8: "statue_area",
    9: "statue_area",
    10: "statue_area",
    11: "terrace_ledge",
    12: "terrace_ledge",
    13: "terrace_ledge",
    14: "terrace_ledge",
    15: "main_hall",
    16: "main_hall",
    17: "main_hall",
    18: "main_hall",
    19: "pool_mushroom",
    20: "grassy_statue",
    21: "portal_area",
    22: "portal_area",
    23: "pool_mushroom",
    24: "pool_mushroom",
    25: "main_hall",
    26: "corridor",
    27: "portal_area",
    28: "statue_area",
    29: "main_hall",
    30: "main_hall",
    31: "main_hall",
    32: "main_hall",
}


# --------------------------------------------------------------------------- #
# 复用现有资产
# --------------------------------------------------------------------------- #
def _import_assets(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    try:
        from backend.avatar_identity import (  # type: ignore
            _context_descriptor,
            appearance_descriptor,
            _similarity,
        )
    except Exception as exc:  # pragma: no cover - 环境异常
        raise SystemExit(f"无法导入 backend 现有资产: {type(exc).__name__}: {exc}")
    return _context_descriptor, appearance_descriptor, _similarity


# --------------------------------------------------------------------------- #
# 统计辅助
# --------------------------------------------------------------------------- #
def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": float("nan"), "median": float("nan"),
                "p90": float("nan"), "max": float("nan")}
    return {
        "n": len(values),
        "min": float(min(values)),
        "median": float(median(values)),
        "p90": float(_pct(values, 0.90)),
        "max": float(max(values)),
    }


# --------------------------------------------------------------------------- #
# 描述子
# --------------------------------------------------------------------------- #
def _read_frame(path: Path):
    import cv2  # type: ignore
    img = cv2.imread(str(path))
    return img


def _compute_vectors(files: list[Path], desc_fn) -> list[tuple[float, ...] | None]:
    vectors = []
    for f in files:
        frame = _read_frame(f)
        if frame is None:
            vectors.append(None)
            continue
        vectors.append(desc_fn(frame))
    return vectors


# --------------------------------------------------------------------------- #
# 核心测量
# --------------------------------------------------------------------------- #
def _analyze(
    vectors: list[tuple[float, ...] | None],
    labels: dict[int, str],
    sim_fn,
) -> dict[str, Any]:
    n = len(vectors)
    valid_indices = [i for i, v in enumerate(vectors) if v is not None]

    intra: list[float] = []
    inter: list[float] = []
    all_pairs: list[float] = []
    consecutive: list[tuple[int, float]] = []

    for i in valid_indices:
        for j in valid_indices:
            if j <= i:
                continue
            s = sim_fn(vectors[i], vectors[j])
            all_pairs.append(s)
            li = labels.get(i + 1)
            lj = labels.get(j + 1)
            if li and lj and li == lj:
                intra.append(s)
            else:
                inter.append(s)

    for i in range(n - 1):
        if vectors[i] is not None and vectors[i + 1] is not None:
            consecutive.append((i, sim_fn(vectors[i], vectors[i + 1])))

    # 区域两两平均相似度矩阵（用各区域帧均值向量）
    region_to_vecs: dict[str, list[tuple[float, ...]]] = {}
    for idx, vec in enumerate(vectors):
        if vec is None:
            continue
        region = labels.get(idx + 1)
        if region is None:
            continue
        region_to_vecs.setdefault(region, []).append(vec)

    region_means: dict[str, tuple[float, ...] | None] = {}
    for r, vecs in region_to_vecs.items():
        if not vecs:
            region_means[r] = None
            continue
        dim = len(vecs[0])
        mean = [sum(float(v[i]) for v in vecs) for i in range(dim)]
        norm = math.sqrt(sum(float(x) ** 2 for x in mean))
        if norm <= 1e-12:
            region_means[r] = None
            continue
        region_means[r] = tuple(float(x) / norm for x in mean)

    region_matrix: dict[str, dict[str, float]] = {}
    for ra, va in region_means.items():
        region_matrix[ra] = {}
        for rb, vb in region_means.items():
            if va is None or vb is None:
                continue
            region_matrix[ra][rb] = sim_fn(va, vb)

    # 可分离性：找出能把 intra 全部保留且 inter 全部排除的阈值区间
    if intra and inter:
        intra_min = min(intra)
        intra_max = max(intra)
        inter_min = min(inter)
        inter_max = max(inter)
        gap = intra_min - inter_max
        separable = gap > 0
    else:
        intra_min = intra_max = inter_min = inter_max = gap = float("nan")
        separable = False

    # 在若干候选阈值下的混淆
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
    threshold_report = []
    for t in thresholds:
        # 把相似度>=t判为"同一地点"
        tp = sum(1 for s in intra if s >= t)
        fn = len(intra) - tp
        fp = sum(1 for s in inter if s >= t)
        tn = len(inter) - fp
        threshold_report.append({
            "threshold": t,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "intra_recall": tp / len(intra) if intra else float("nan"),
            "inter_specificity": tn / len(inter) if inter else float("nan"),
        })

    return {
        "n_frames": n,
        "valid_frames": len(valid_indices),
        "labels": labels,
        "regions": {r: len(v) for r, v in region_to_vecs.items()},
        "all_pairs": _summarize(all_pairs),
        "consecutive": {
            "min": float(min((s for _, s in consecutive), default=float("nan"))),
            "median": float(median([s for _, s in consecutive]) if consecutive else float("nan")),
        },
        "intra_region": _summarize(intra),
        "inter_region": _summarize(inter),
        "separability": {
            "intra_min": intra_min,
            "intra_max": intra_max,
            "inter_min": inter_min,
            "inter_max": inter_max,
            "gap": gap,
            "cleanly_separable": separable,
        },
        "threshold_grid": threshold_report,
        "region_mean_similarity_matrix": region_matrix,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="地点可区分性离线测量")
    parser.add_argument("--frames-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "tmp" / "keyframes",
                        help="关键帧目录（默认 tmp/keyframes）")
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[2],
                        help="仓库根目录")
    parser.add_argument("--method", choices=["context", "color"], default="context",
                        help="指纹方法")
    parser.add_argument("--labels-json", type=Path, default=None,
                        help="自定义区域标注 JSON（帧索引1-based -> 区域名）")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="输出 JSON 报告路径")
    args = parser.parse_args()

    ctx_fn, app_fn, sim_fn = _import_assets(args.repo)
    desc_fn = ctx_fn if args.method == "context" else (lambda f: app_fn(f, (0.0, 0.0, 1.0, 1.0)))

    if args.labels_json:
        labels = {int(k): v for k, v in json.loads(args.labels_json.read_text(encoding="utf-8")).items()}
    else:
        labels = DEFAULT_LABELS

    files = sorted(args.frames_dir.glob("frame_*.jpg"))
    if not files:
        print(f"未在 {args.frames_dir} 找到 frame_*.jpg", file=sys.stderr)
        return 1

    # 只保留有标注的帧
    files = [f for f in files if int("".join(filter(str.isdigit, f.stem)) or -1) in labels]
    if not files:
        print("没有帧与标注匹配", file=sys.stderr)
        return 1

    vectors = _compute_vectors(files, desc_fn)
    report = _analyze(vectors, labels, sim_fn)
    report["method"] = args.method
    report["frames_dir"] = str(args.frames_dir)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"\n[报告已写入 {args.json_out}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
