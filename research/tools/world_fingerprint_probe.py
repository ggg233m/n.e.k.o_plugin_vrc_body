"""离线「出生点视觉指纹」可行性探针（不接实时感知链路）。

目的
----
验证一个设想：VRChat 每个世界的出生点画面是独特的，能否用视觉指纹自动识别世界，
从而省掉手动输入 world 标识。本脚本只做**离线测量**，不动 navigator/vision/service
的运行时逻辑。

它**优先复用**现有资产，而不是重新实现：
  * ``backend.avatar_identity._context_descriptor`` —— 4x8 网格低分辨率背景指纹，
    L2 归一化，点积即余弦相似度。原设计用于「剔除人物框后的背景」，最接近场景指纹。
  * ``backend.avatar_identity.appearance_descriptor`` —— 颜色直方图外观描述子。
  * ``backend.avatar_identity._similarity`` —— 余弦相似度（对长度不匹配返回 0）。
  * ``backend.reid_embedder.OsnetReidEmbedder`` —— OSNet ONNX 行人重识别嵌入。
    注意：它是**行人**重识别网络，用在整帧场景上效果未知，且依赖 onnxruntime；
    本机若缺 onnxruntime，``available`` 为 False，脚本会如实跳过，不伪造结果。

输入
----
  * ``video`` 子命令：视频文件，按 fps/时间段抽帧（解码交给 ffmpeg，AV1 解出最稳）。
  * ``images`` 子命令：图片目录（将来用户手动采集出生点素材用）。

输出
----
  * 相似度矩阵 + 统计：类内相似度分布、类间相似度分布（若素材含多世界）、
    能否用单一阈值分离、重叠区间。
  * 一份 ROADMAP 风格的严格报告（n / 中位数 / p90 / max / 是否越过候选阈值）。
  * 可选把每个聚类代表帧落盘，方便人工核验「聚类是不是真的对应不同世界」。

用法示例
--------
  python research/tools/world_fingerprint_probe.py video "2026-09-18 07-31-11.mkv" \
      --fps 2 --method context --threshold 0.90 --save-clusters tmp/clusters

  python research/tools/world_fingerprint_probe.py images ./my_spawn_frames \
      --method context --threshold 0.90
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from typing import Any, Sequence


# --------------------------------------------------------------------------- #
# 复用现有资产（不修改它们，仅导入）。                                       #
# --------------------------------------------------------------------------- #
def _import_assets(repo_root: Path):
    """导入 backend 的现有指纹函数；失败则如实报错退出。"""
    sys.path.insert(0, str(repo_root))
    try:
        from backend.avatar_identity import (  # type: ignore
            _context_descriptor,
            appearance_descriptor,
            _similarity,
        )
        from backend.reid_embedder import OsnetReidEmbedder  # type: ignore
    except Exception as exc:  # pragma: no cover - 环境异常
        raise SystemExit(f"无法导入 backend 现有资产: {type(exc).__name__}: {exc}")
    return _context_descriptor, appearance_descriptor, _similarity, OsnetReidEmbedder


# --------------------------------------------------------------------------- #
# 帧获取                                                                      #
# --------------------------------------------------------------------------- #
def extract_frames_ffmpeg(video: Path, fps: float, start_s: float, end_s: float,
                          out_dir: Path) -> list[Path]:
    """用 ffmpeg 抽帧（AV1 等格式 cv2 未必能解，ffmpeg 最稳）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s}"]
    cmd += ["-i", str(video)]
    if end_s > 0:
        cmd += ["-to", f"{end_s}"]
    cmd += ["-vf", vf, "-q:v", "3", str(out_dir / "frame_%06d.jpg")]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("ffmpeg 未安装或不在 PATH，无法抽帧")
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg 抽帧失败: {exc.stderr[:500]}")
    files = sorted(out_dir.glob("frame_*.jpg"), key=lambda p: p.stat().st_mtime)
    return files


def load_images(directory: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: list[Path] = []
    for ext in exts:
        files.extend(directory.glob(ext))
    files = sorted({f for f in files if f.is_file()})
    if not files:
        raise SystemExit(f"目录 {directory} 下没有图片文件")
    return files


# --------------------------------------------------------------------------- #
# 描述子计算                                                                  #
# --------------------------------------------------------------------------- #
def make_descriptor_fn(method: str, ctx_fn, app_fn, osnet):
    """返回一个 (frame_bgr -> vector|None) 的函数。"""
    if method == "context":
        # 整帧背景指纹。原设计为剔除人物框；此处默认不剔除（离线素材无检测框），
        # 在报告中说明该限制，并保留 excluded_bboxes 扩展位。
        def fn(frame):
            return ctx_fn(frame)
        return fn
    if method == "color":
        def fn(frame):
            return app_fn(frame, (0.0, 0.0, 1.0, 1.0))
        return fn
    if method == "osnet":
        if osnet is None or not getattr(osnet, "available", False):
            return None
        def fn(frame):
            return osnet.embed(frame, (0.0, 0.0, 1.0, 1.0))
        return fn
    raise SystemExit(f"未知 method: {method}")


# --------------------------------------------------------------------------- #
# 聚类 / 统计                                                                 #
# --------------------------------------------------------------------------- #
def pct(values: Sequence[float], q: float) -> float:
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


def greedy_cluster(vectors: list, sim_fn, threshold: float) -> list[list[int]]:
    """按帧序贪心聚类：每帧归入首个代表相似度 ≥ threshold 的簇，否则新簇。"""
    clusters: list[list[int]] = []

    def rep_sim(cluster: list[int], vec) -> float:
        best = -1.0
        for idx in cluster:
            s = sim_fn(vectors[idx], vec)
            if s > best:
                best = s
        return best

    for i, vec in enumerate(vectors):
        if vec is None:
            continue
        placed = False
        for cl in clusters:
            if rep_sim(cl, vec) >= threshold:
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": float("nan"), "median": float("nan"),
                "p90": float("nan"), "max": float("nan")}
    return {
        "n": len(values),
        "min": float(min(values)),
        "median": float(median(values)),
        "p90": float(pct(values, 0.90)),
        "max": float(max(values)),
    }


def analyze(vectors: list, sim_fn, threshold: float) -> dict[str, Any]:
    """完整的相似度矩阵统计 + 聚类。"""
    n = len(vectors)
    # 全相似度矩阵（上三角）
    matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if vectors[i] is None or vectors[j] is None:
                matrix[i][j] = None
                continue
            matrix[i][j] = sim_fn(vectors[i], vectors[j])

    def all_pairs() -> list[float]:
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                v = matrix[i][j]
                if v is not None:
                    out.append(v)
        return out

    pairs = all_pairs()

    # 帧序相邻相似度（检测世界切换的谷点）
    consecutive = []
    for i in range(n - 1):
        if vectors[i] is not None and vectors[i + 1] is not None:
            consecutive.append((i, sim_fn(vectors[i], vectors[i + 1])))

    # 聚类
    clusters = greedy_cluster(vectors, sim_fn, threshold)
    # 簇间：不同簇两两最小相似度（最能暴露跨世界分离度）
    inter = []
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            ca, cb = clusters[a], clusters[b]
            best = -1.0
            for ia in ca:
                for ib in cb:
                    if vectors[ia] is not None and vectors[ib] is not None:
                        s = sim_fn(vectors[ia], vectors[ib])
                        if s > best:
                            best = s
            if best >= 0:
                inter.append(best)
    # 类内：同簇两两相似度
    intra = []
    for cl in clusters:
        for x in range(len(cl)):
            for y in range(x + 1, len(cl)):
                ia, ib = cl[x], cl[y]
                if vectors[ia] is not None and vectors[ib] is not None:
                    intra.append(sim_fn(vectors[ia], vectors[ib]))

    return {
        "n_frames": n,
        "valid_frames": sum(1 for v in vectors if v is not None),
        "pairs_total": len(pairs),
        "all_pair_similarity": summarize(pairs),
        "consecutive_min": float(min((s for _, s in consecutive), default=float("nan"))),
        "consecutive_valleys": [
            {"frame_index": i, "sim": round(s, 4)}
            for i, s in consecutive if s < threshold
        ],
        "clusters": [{"size": len(c), "frame_range": [c[0], c[-1]]} for c in clusters],
        "n_clusters": len(clusters),
        "intra_cluster_similarity": summarize(intra),
        "inter_cluster_similarity": summarize(inter),
    }


# --------------------------------------------------------------------------- #
# 主流程                                                                      #
# --------------------------------------------------------------------------- #
def read_frame(path: Path):
    import cv2  # 懒导入，缺 cv2 时只在这一步报错
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img  # BGR


def run(files: list[Path], method: str, threshold: float,
        save_clusters: Path | None, ctx_fn, app_fn, osnet, sim_fn) -> dict[str, Any]:
    desc_fn = make_descriptor_fn(method, ctx_fn, app_fn, osnet)
    if desc_fn is None:
        # osnet 不可用：如实报告，不伪造
        return {
            "method": method,
            "osnet_available": False,
            "error": "OSNet 嵌入不可用（缺 onnxruntime 或模型），按项目约束报 available=false，未伪造。",
        }

    vectors = []
    for f in files:
        frame = read_frame(f)
        if frame is None:
            vectors.append(None)
            continue
        vectors.append(desc_fn(frame))

    report = analyze(vectors, sim_fn, threshold)
    report["method"] = method
    report["threshold"] = threshold
    report["osnet_available"] = bool(osnet and getattr(osnet, "available", False))

    # 候选阈值可分离性评估
    intra = report["intra_cluster_similarity"]
    inter = report["inter_cluster_similarity"]
    if inter["n"] > 0:
        # 跨世界相似度中有多少「高过」候选阈值（误判为同世界）
        # 需要回到原始 inter 列表；这里用简化结论：给出 max 与阈值比较
        report["separability"] = {
            "intra_max": intra["max"],
            "inter_max": inter["max"],
            "intra_min": intra["min"],
            "inter_min": inter["min"],
            "gap": (intra["min"] if intra["n"] else float("nan")) - inter["max"],
            "note": "带间隙>0 表示可用单一阈值分离；重叠则不可靠。",
        }
    else:
        report["separability"] = {
            "note": "仅 1 个聚类（≥2 个世界才能量到类间分离度）；本次素材无法判定跨世界分离。",
        }

    # 落盘聚类代表帧
    if save_clusters is not None:
        save_clusters.mkdir(parents=True, exist_ok=True)
        for ci, cl in enumerate(greedy_cluster(vectors, sim_fn, threshold)):
            if not cl:
                continue
            # 选簇内与第一帧相似度最高的作为代表
            rep = cl[0]
            rep_vec = vectors[rep]
            best_s = 2.0
            for idx in cl:
                if vectors[idx] is None:
                    continue
                s = sim_fn(rep_vec, vectors[idx])
                if abs(1.0 - s) < best_s:
                    best_s = abs(1.0 - s)
                    rep = idx
            src = files[rep]
            import shutil
            shutil.copy(src, save_clusters / f"cluster_{ci:02d}_rep.jpg")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="离线出生点视觉指纹可行性探针")
    sub = parser.add_subparsers(dest="source", required=True)

    repo_default = Path(__file__).resolve().parents[2]
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", type=Path, default=repo_default,
                        help="仓库根目录（用于导入 backend 资产）")
    common.add_argument("--method", choices=["context", "color", "osnet"],
                        default="context", help="指纹方法")
    common.add_argument("--threshold", type=float, default=0.90,
                        help="候选聚类/匹配阈值（参考 avatar_identity 现有值）")
    common.add_argument("--save-clusters", type=Path, default=None,
                        help="把每个聚类代表帧落盘到该目录，便于人工核验")
    common.add_argument("--json-out", type=Path, default=None,
                        help="把报告写成 JSON 文件")

    v = sub.add_parser("video", parents=[common], help="从视频抽帧")
    v.add_argument("path", type=Path, help="视频文件")
    v.add_argument("--fps", type=float, default=2.0, help="抽帧率")
    v.add_argument("--start", type=float, default=0.0, help="起始秒")
    v.add_argument("--end", type=float, default=0.0, help="结束秒（0=到结尾）")

    img = sub.add_parser("images", parents=[common], help="从图片目录")
    img.add_argument("directory", type=Path, help="图片目录")

    args = parser.parse_args()

    ctx_fn, app_fn, sim_fn, OsnetReidEmbedder = _import_assets(args.repo)

    # OSNet 仅在 method=osnet 时尝试初始化（避免无谓依赖）
    osnet = None
    if args.method == "osnet":
        # 尝试定位模型；找不到则 available=False
        model_candidates = list(args.repo.glob("models/**/*osnet*.onnx"))
        model_candidates += list(args.repo.glob("**/osnet*.onnx"))
        mp = model_candidates[0] if model_candidates else Path("models/osnet_x0_25.onnx")
        osnet = OsnetReidEmbedder(model_path=mp)

    if args.source == "video":
        with tempfile.TemporaryDirectory() as td:
            files = extract_frames_ffmpeg(args.path, args.fps, args.start, args.end, Path(td))
            if not files:
                print("ffmpeg 未抽出任何帧", file=sys.stderr)
                return 1
            report = run(files, args.method, args.threshold, args.save_clusters,
                         ctx_fn, app_fn, osnet, sim_fn)
    else:
        files = load_images(args.directory)
        report = run(files, args.method, args.threshold, args.save_clusters,
                     ctx_fn, app_fn, osnet, sim_fn)

    report["n_input_frames"] = len(files)
    report["source"] = args.source

    printed = {k: v for k, v in report.items() if k != "clusters"} if False else report
    print(json.dumps(printed, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"\n[报告已写入 {args.json_out}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
