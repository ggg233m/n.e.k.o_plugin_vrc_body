"""世界出生点视觉指纹的离线可行性测量工具。

回答一个问题：VRChat 世界出生点的画面指纹，"同一世界内"（类内）与
"不同世界之间"（类间）的余弦相似度能否被一个阈值干净分开。

刻意只做离线测量，不接入任何实时感知循环。指纹计算复用运行时已验证的
实现：``backend.avatar_identity._context_descriptor``（4×8 网格背景指纹，
L2 归一化，点积即余弦相似度）与 ``backend.reid_embedder.OsnetReidEmbedder``
（OSNet 行人重识别网络，用在整帧场景上属于未验证用法，输出仅供参考）。

用法：
    # 视频：顺序解码（AV1 等编码上 seek 极慢，绝不要用 CAP_PROP_POS_FRAMES）
    python research/tools/measure_world_fingerprint.py --video 录屏.mkv --sample-fps 1.0

    # 图片目录：每个子目录算一个世界；扁平目录算单一世界
    python research/tools/measure_world_fingerprint.py --images 采集目录/

    # 显式标注：JSON 文件 {"文件名或相对路径": "世界名", ...}
    python research/tools/measure_world_fingerprint.py --images 目录/ --labels 标注.json

素材只有一个世界时，类间分离度**无法测量**，工具会如实输出
"INTER-CLASS NOT MEASURABLE"，不会用任何假设或模拟数据补数。

统计口径遵循 ROADMAP 的教训：报 n、中位数、p10/p90、min/max，并对照
运行时现有阈值（avatar_identity._CONTEXT_MATCH_THRESHOLD = 0.90），
而不是凭感觉宣布某个阈值"应该可行"。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.avatar_identity import _context_descriptor  # noqa: E402

_DEFAULT_OSNET_MODEL = _REPO_ROOT / "models" / "osnet_x0_25_msmt17" / "osnet_x0_25_msmt17_dynamic.onnx"
_RUNTIME_CONTEXT_MATCH_THRESHOLD = 0.90  # avatar_identity._CONTEXT_MATCH_THRESHOLD

# 视频模式下的时间差分桶（秒）：近邻帧视野重叠大，远端帧才接近"重进世界
# 但视角漂移"的真实考验。
_TIME_BUCKETS = ((0.0, 2.0, "<2s"), (2.0, 10.0, "2-10s"), (10.0, float("inf"), ">=10s"))


def _load_frames_from_video(
    path: Path, sample_fps: float, start: float, end: float, max_frames: int
) -> tuple[list, list[float]]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / max(sample_fps, 1e-6))))
    start_index = max(0, int(start * fps))
    end_index = None if end <= 0 else int(end * fps)
    frames: list = []
    timestamps: list[float] = []
    index = 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index >= start_index and (index - start_index) % step == 0:
            if end_index is not None and index > end_index:
                break
            ok, frame = capture.retrieve()
            if ok and frame is not None:
                frames.append(frame)
                timestamps.append(index / fps)
                if len(frames) >= max_frames:
                    break
        index += 1
    capture.release()
    if not frames:
        raise SystemExit("no frames decoded from video")
    return frames, timestamps


def _load_images(directory: Path, labels_path: Path | None, recursive: bool, max_frames: int):
    import cv2

    pattern = "**/*" if recursive else "*"
    files = sorted(
        item
        for item in directory.glob(pattern)
        if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} and item.is_file()
    )
    if not files:
        raise SystemExit(f"no images under: {directory}")
    explicit = None
    if labels_path is not None:
        explicit = json.loads(labels_path.read_text(encoding="utf-8"))
    frames = []
    labels = []
    for item in files:
        if explicit is not None:
            key_variants = (str(item.relative_to(directory)).replace("\\", "/"), item.name)
            label = next((explicit[key] for key in key_variants if key in explicit), None)
            if label is None:
                continue
        else:
            parent = item.parent.relative_to(directory)
            label = str(parent) if str(parent) != "." else directory.name
        frame = cv2.imread(str(item), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frames.append(frame)
        labels.append(str(label))
        if len(frames) >= max_frames:
            break
    if not frames:
        raise SystemExit("no decodable images matched the labels")
    return frames, labels


def _build_fingerprints(frames, methods, osnet_model: Path):
    """返回 {method: (向量矩阵, 错误信息或 None)}。"""
    results: dict[str, tuple | None] = {}
    if "context" in methods:
        vectors = []
        for frame in frames:
            descriptor = _context_descriptor(frame)
            vectors.append(None if descriptor is None else list(descriptor))
        results["context"] = _stack(vectors)
    if "osnet" in methods:
        error = None
        try:
            from backend.reid_embedder import OsnetReidEmbedder

            embedder = OsnetReidEmbedder(model_path=osnet_model)
            if not embedder.available:
                error = f"OSNet unavailable: {embedder.status().get('error')}"
            else:
                vectors = []
                for frame in frames:
                    vector = embedder.embed(frame, (0.0, 0.0, 1.0, 1.0))
                    vectors.append(None if vector is None else list(vector))
                results["osnet"] = _stack(vectors)
        except Exception as exc:  # 缺 onnxruntime 等属预期降级
            error = f"{type(exc).__name__}: {exc}"
        if error is not None:
            results["osnet"] = (None, error)
    return {name: value for name, value in results.items() if value is not None}


def _stack(vectors):
    import numpy as np

    kept = [index for index, vector in enumerate(vectors) if vector is not None]
    if not kept:
        return None, "all fingerprints failed"
    matrix = np.asarray([vectors[index] for index in kept], dtype=np.float32)
    return (matrix, kept), None


def _pair_stats(similarities) -> dict:
    import numpy as np

    values = np.asarray(similarities, dtype=np.float64)
    if values.size == 0:
        return {"n_pairs": 0}
    return {
        "n_pairs": int(values.size),
        "mean": round(float(values.mean()), 4),
        "median": round(float(np.median(values)), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
    }


def _separation(intra_values, inter_values) -> dict:
    import numpy as np

    intra = np.asarray(intra_values, dtype=np.float64)
    inter = np.asarray(inter_values, dtype=np.float64)
    candidates = np.unique(np.concatenate([intra, inter]))
    midpoints = (candidates[:-1] + candidates[1:]) / 2.0
    best = {"threshold": None, "balanced_accuracy": 0.0}
    n_inter = max(1, inter.size)
    n_intra = max(1, intra.size)
    for threshold in midpoints:
        tpr = float((intra >= threshold).sum()) / n_intra
        tnr = float((inter < threshold).sum()) / n_inter
        score = (tpr + tnr) / 2.0
        if score > best["balanced_accuracy"]:
            best = {"threshold": round(float(threshold), 4), "balanced_accuracy": round(score, 4)}
    # Mann-Whitney AUC：类内分数高于类间的概率（含并列按 0.5 计）。
    order = np.argsort(np.concatenate([inter, intra]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    combined = np.concatenate([inter, intra])
    for value in np.unique(combined):
        mask = combined == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    r_sum = ranks[inter.size :].sum()
    auc = (r_sum - n_intra * (n_intra + 1) / 2.0) / (n_intra * n_inter)
    strict_gap = round(float(intra.min() - inter.max()), 4) if intra.size and inter.size else None
    return {
        "best_threshold": best["threshold"],
        "best_balanced_accuracy": best["balanced_accuracy"],
        "auc_intra_gt_inter": round(float(auc), 4),
        "strict_gap_intra_min_minus_inter_max": strict_gap,
        "strictly_separable": bool(strict_gap is not None and strict_gap > 0.0),
    }


def _temporal_decay(similarities, timestamps, pair_index) -> dict:
    buckets: dict[str, list[float]] = {name: [] for _, _, name in _TIME_BUCKETS}
    for (i, j), value in zip(pair_index, similarities):
        delta = abs(timestamps[i] - timestamps[j])
        for low, high, name in _TIME_BUCKETS:
            if low <= delta < high:
                buckets[name].append(float(value))
                break
    return {name: _pair_stats(values) for name, values in buckets.items()}


def _analyze(method: str, matrix, kept, labels, timestamps, is_video: bool) -> dict:
    import numpy as np

    kept_labels = [labels[index] for index in kept]
    kept_timestamps = [timestamps[index] for index in kept] if timestamps else None
    similarity = matrix @ matrix.T
    count = matrix.shape[0]
    pair_index = []
    intra_values = []
    inter_values = []
    for i in range(count):
        for j in range(i + 1, count):
            value = float(similarity[i, j])
            pair_index.append((i, j))
            if kept_labels[i] == kept_labels[j]:
                intra_values.append(value)
            else:
                inter_values.append(value)
    report: dict = {"method": method, "n_frames": count}
    report["intra_class"] = _pair_stats(intra_values)
    over_runtime = sum(1 for value in intra_values if value < _RUNTIME_CONTEXT_MATCH_THRESHOLD)
    report["intra_pairs_below_runtime_0.90"] = f"{over_runtime}/{len(intra_values)}"
    if inter_values:
        report["inter_class"] = _pair_stats(inter_values)
        report["separation"] = _separation(intra_values, inter_values)
    else:
        report["inter_class"] = "INTER-CLASS NOT MEASURABLE: input contains only one world label"
    if is_video and kept_timestamps is not None and len(intra_values) > 0:
        report["intra_temporal_decay"] = _temporal_decay(intra_values, kept_timestamps, pair_index)
    return report


def _print_report(analyses: dict, input_summary: dict) -> None:
    print(f"input: {json.dumps(input_summary, ensure_ascii=False)}")
    print(f"runtime reference: context match threshold = {_RUNTIME_CONTEXT_MATCH_THRESHOLD}")
    for method, report in analyses.items():
        print(f"\n=== method: {method} ===")
        print(f"  frames fingerprinted: {report['n_frames']}")
        intra = report["intra_class"]
        if intra.get("n_pairs", 0) == 0:
            print("  intra-class: no pairs")
        else:
            print(
                "  intra-class (same world): n={n_pairs} mean={mean} median={median} "
                "p10={p10} p90={p90} min={min} max={max}".format(**intra)
            )
            print(
                f"  intra pairs below runtime { _RUNTIME_CONTEXT_MATCH_THRESHOLD }: "
                f"{report['intra_pairs_below_runtime_0.90']}"
            )
        if isinstance(report.get("inter_class"), str):
            print(f"  inter-class (cross world): {report['inter_class']}")
        else:
            inter = report["inter_class"]
            print(
                "  inter-class (cross world): n={n_pairs} mean={mean} median={median} "
                "p10={p10} p90={p90} min={min} max={max}".format(**inter)
            )
            sep = report["separation"]
            print(
                f"  separation: best_threshold={sep['best_threshold']} "
                f"balanced_acc={sep['best_balanced_accuracy']} auc={sep['auc_intra_gt_inter']} "
                f"strict_gap={sep['strict_gap_intra_min_minus_inter_max']} "
                f"strictly_separable={sep['strictly_separable']}"
            )
        decay = report.get("intra_temporal_decay")
        if decay:
            parts = []
            for name, stats in decay.items():
                parts.append(f"{name}: median={stats.get('median')} (n={stats.get('n_pairs')})")
            print(f"  intra temporal decay: {'; '.join(parts)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="video file to sample")
    source.add_argument("--images", type=Path, help="image directory (subdirectory = world label)")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="video sampling rate (default 1.0)")
    parser.add_argument("--start", type=float, default=0.0, help="video start seconds")
    parser.add_argument("--end", type=float, default=0.0, help="video end seconds (0 = till the end)")
    parser.add_argument("--world-label", type=str, default=None, help="single-world label for --video")
    parser.add_argument("--labels", type=Path, default=None, help="JSON {filename: world} for --images")
    parser.add_argument("--recursive", action="store_true", help="search images recursively")
    parser.add_argument("--max-frames", type=int, default=400, help="sampling cap (default 400)")
    parser.add_argument("--methods", type=str, default="context,osnet")
    parser.add_argument("--osnet-model", type=Path, default=_DEFAULT_OSNET_MODEL)
    parser.add_argument("--out-json", type=Path, default=None, help="write full report JSON here")
    args = parser.parse_args(argv)

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if args.video is not None:
        frames, timestamps = _load_frames_from_video(
            args.video, args.sample_fps, args.start, args.end, args.max_frames
        )
        label = args.world_label or args.video.stem
        labels = [label] * len(frames)
        is_video = True
        input_summary = {
            "source": str(args.video),
            "worlds": [label],
            "frames_sampled": len(frames),
            "sample_fps": args.sample_fps,
        }
    else:
        frames, labels = _load_images(args.images, args.labels, args.recursive, args.max_frames)
        timestamps = None
        is_video = False
        input_summary = {
            "source": str(args.images),
            "worlds": sorted(set(labels)),
            "frames_sampled": len(frames),
        }
    print(f"world labels present: {sorted(set(labels))} (counts: "
          f"{ {name: labels.count(name) for name in sorted(set(labels))} })")

    built = _build_fingerprints(frames, methods, args.osnet_model)
    if not built:
        raise SystemExit("no fingerprint method produced results")
    analyses = {}
    for method, (payload, error) in built.items():
        if payload is None:
            print(f"=== method: {method} ===\n  SKIPPED: {error}")
            continue
        matrix, kept = payload
        if matrix.shape[0] < 2:
            print(f"=== method: {method} ===\n  SKIPPED: fewer than 2 usable frames")
            continue
        analyses[method] = _analyze(method, matrix, kept, labels, timestamps or [], is_video)
    _print_report(analyses, input_summary)
    if args.out_json is not None:
        payload = {"input": input_summary, "runtime_threshold": _RUNTIME_CONTEXT_MATCH_THRESHOLD}
        for method, report in analyses.items():
            payload[method] = report
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport written: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
