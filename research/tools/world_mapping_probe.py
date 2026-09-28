"""离线「世界模型增量建图」可行性探针（不接实时链路）。

目的
----
回答一个 go/no-go 问题：给定一段玩家在**单个世界**里真实走动的录屏，
用现有指纹方法做**流式增量聚类**（边走边建节点），能否建出一张有意义的
拓扑图（地点节点 + 节点间转移边 + 重访识别）。

与 ``world_fingerprint_probe.py``（一次性全局聚类，回答"是不是同一世界"）
不同，本脚本模拟真实建图的在线过程：

  1. 按时间顺序逐帧取指纹；
  2. 与已有节点比对（帧向量 vs 节点内所有成员的最大相似度）；
  3. 相似度 >= 阈值 → 归入该节点；所有节点都不匹配 → 创建新节点；
  4. 记录相邻帧的节点转移 → 边。

评估维度（项目验收口径）：
  * 节点数：塌成 1 = 没建图；每帧一个 = 没聚类。65 秒室内游走合理区间 2~8。
  * 时间局部性：相邻帧应落在同一节点（转移不应疯狂跳）。
  * 重访识别：走过回头路时能否匹配回之前的节点（拓扑图可复用的关键）。
  * 阈值敏感性：多个阈值下节点数是否稳定。

复用资产（不修改）：
  * ``backend.avatar_identity._context_descriptor`` / ``appearance_descriptor``
  * ``backend.avatar_identity._similarity``
  * ``backend.reid_embedder.OsnetReidEmbedder``（可选，整帧用法未经验证）

已知限制（如实记录）：离线素材没有检测框，无法剔除玩家自身身体/手持物；
录屏末尾若打开过 VRChat 菜单，菜单会污染指纹。

用法
----
  .venv/Scripts/python.exe research/tools/world_mapping_probe.py images .tmp \
      --pattern wm_f_ --methods context,color,osnet \
      --thresholds 0.70,0.75,0.80,0.85,0.90,0.95 \
      --json-out .tmp/world_mapping_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Callable, Sequence


def _import_assets(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    try:
        from backend.avatar_identity import (  # type: ignore
            _context_descriptor,
            appearance_descriptor,
            _similarity,
        )
        from backend.reid_embedder import OsnetReidEmbedder  # type: ignore
    except Exception as exc:
        raise SystemExit(f"无法导入 backend 现有资产: {type(exc).__name__}: {exc}")
    return _context_descriptor, appearance_descriptor, _similarity, OsnetReidEmbedder


def load_frames(directory: Path, pattern: str) -> list[Path]:
    files = sorted(directory.glob(f"{pattern}*.jpg"))
    if not files:
        raise SystemExit(f"{directory} 下没有 {pattern}*.jpg")
    return files


def read_frame(path: Path):
    import cv2  # 懒导入
    img = cv2.imread(str(path))
    return img  # BGR；失败为 None


# --------------------------------------------------------------------------- #
# 增量（流式）聚类                                                             #
# --------------------------------------------------------------------------- #
def incremental_map(
    vectors: list,
    sim_fn: Callable[[Sequence[float], Sequence[float]], float],
    threshold: float,
) -> dict[str, Any]:
    """模拟边走边建图：逐帧与已有节点比对，匹配则归入，否则新建节点。

    返回节点表、每帧的节点指派、以及相邻帧转移（边，带计数）。
    """
    nodes: list[list[int]] = []  # 每个节点存成员帧下标
    assignment: list[int | None] = []
    edges: dict[tuple[int, int], int] = {}
    transitions = 0
    for i, vec in enumerate(vectors):
        if vec is None:
            assignment.append(None)
            continue
        best_node, best_sim = -1, -1.0
        for ni, members in enumerate(nodes):
            for idx in members:
                s = sim_fn(vectors[idx], vec)
                if s > best_sim:
                    best_sim, best_node = s, ni
        if best_sim >= threshold:
            nodes[best_node].append(i)
            assignment.append(best_node)
        else:
            nodes.append([i])
            assignment.append(len(nodes) - 1)
        # 记录与上一有效帧的转移
        prev = next((assignment[j] for j in range(i - 1, -1, -1)
                     if assignment[j] is not None), None)
        cur = assignment[i]
        if prev is not None and cur is not None:
            if prev != cur:
                transitions += 1
            key = (min(prev, cur), max(prev, cur))
            edges[key] = edges.get(key, 0) + 1
    return {
        "n_nodes": len(nodes),
        "node_sizes": [len(m) for m in nodes],
        "node_frame_ranges": [[m[0], m[-1]] for m in nodes],
        "assignment": assignment,
        "edges": [{"a": a, "b": b, "count": c} for (a, b), c in
                  sorted(edges.items())],
        "n_transitions": transitions,
    }


def locality_stats(assignment: list[int | None]) -> dict[str, float]:
    """时间局部性：相邻有效帧对中落在同一节点的比例 + 平均连续驻留长度。"""
    same = total = 0
    runs: list[int] = []
    run = 0
    prev: int | None = None
    for a in assignment:
        if a is None:
            continue
        if prev is not None:
            total += 1
            if a == prev:
                same += 1
                run += 1
            else:
                if run:
                    runs.append(run)
                run = 1
        else:
            run = 1
        prev = a
    if run:
        runs.append(run)
    return {
        "adjacent_pairs": total,
        "same_node_ratio": (same / total) if total else float("nan"),
        "n_runs": len(runs),
        "mean_run_len": (sum(runs) / len(runs)) if runs else float("nan"),
    }


def revisit_report(assignment: list[int | None], n_nodes: int) -> dict[str, Any]:
    """重访识别：每个节点的成员帧是否在时间上分裂成多段（分裂=检测到重访）。"""
    out = []
    for ni in range(n_nodes):
        frames = [i for i, a in enumerate(assignment) if a == ni]
        segments = []
        start = prev = frames[0]
        for f in frames[1:]:
            if f != prev + 1:
                segments.append((start, prev))
                start = f
            prev = f
        segments.append((start, prev))
        out.append({"node": ni, "size": len(frames),
                    "segments": [list(s) for s in segments],
                    "revisited": len(segments) > 1})
    return {
        "nodes": out,
        "n_revisited_nodes": sum(1 for o in out if o["revisited"]),
    }


def summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": float("nan"), "median": float("nan"),
                "p90": float("nan"), "max": float("nan")}
    s = sorted(values)
    return {"n": len(s), "min": s[0], "median": median(s),
            "p90": s[int(0.9 * (len(s) - 1))], "max": s[-1]}


# --------------------------------------------------------------------------- #
# 主流程                                                                      #
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="离线世界模型增量建图探针")
    sub = parser.add_subparsers(dest="source", required=True)
    repo_default = Path(__file__).resolve().parents[2]
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", type=Path, default=repo_default)
    common.add_argument("--methods", default="context,color",
                        help="逗号分隔: context,color,osnet")
    common.add_argument("--thresholds", default="0.70,0.75,0.80,0.85,0.90,0.95")
    common.add_argument("--json-out", type=Path, default=None)

    img = sub.add_parser("images", parents=[common], help="图片目录（按文件名序=时间序）")
    img.add_argument("directory", type=Path)
    img.add_argument("--pattern", default="frame_", help="文件名前缀")

    args = parser.parse_args()
    ctx_fn, app_fn, sim_fn, OsnetReidEmbedder = _import_assets(args.repo)

    files = load_frames(args.directory, args.pattern)
    frames = []
    for f in files:
        im = read_frame(f)
        if im is None:
            print(f"[warn] 读图失败: {f}", file=sys.stderr)
        frames.append(im)
    valid = [f for f in frames if f is not None]
    print(f"载入 {len(valid)}/{len(frames)} 帧", file=sys.stderr)

    thresholds = [float(t) for t in args.thresholds.split(",")]
    report: dict[str, Any] = {
        "n_frames": len(frames),
        "n_valid_frames": len(valid),
        "thresholds": thresholds,
        "limitations": [
            "离线素材无检测框，玩家自身身体/手持物未从指纹中剔除（原设计 _context_descriptor 支持 excluded_bboxes，实时链路可剔除）",
            "录屏若含 VRChat 菜单/特效遮挡，会污染指纹",
            "_context_descriptor 原设计目标是'同一视角'判断，不是地点识别",
        ],
        "methods": {},
    }

    osnet = None
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        if method == "context":
            def desc(frame, _f=ctx_fn):
                return _f(frame)
        elif method == "color":
            def desc(frame, _f=app_fn):
                return _f(frame, (0.0, 0.0, 1.0, 1.0))
        elif method == "osnet":
            if osnet is None:
                cands = list(args.repo.glob("models/**/*osnet*.onnx"))
                osnet = OsnetReidEmbedder(
                    model_path=cands[0]) if cands else OsnetReidEmbedder(
                    model_path=args.repo / "models" / "osnet_x0_25.onnx")
            if not getattr(osnet, "available", False):
                report["methods"]["osnet"] = {
                    "available": False,
                    "note": "OSNet 不可用（缺 onnxruntime/模型），如实报告，未伪造"}
                continue

            def desc(frame, _o=osnet):
                return _o.embed(frame, (0.0, 0.0, 1.0, 1.0))
        else:
            raise SystemExit(f"未知 method: {method}")

        vectors = [desc(f) if f is not None else None for f in frames]

        # 首帧 vs 其余帧的相似度曲线：量化空间变化（素材合格性的数值佐证）
        base = next((v for v in vectors if v is not None), None)
        sim_curve = [sim_fn(base, v) if (base is not None and v is not None)
                     else None for v in vectors]
        sims_valid = [s for s in sim_curve if s is not None]
        consec = [sim_fn(vectors[i], vectors[i + 1])
                  for i in range(len(vectors) - 1)
                  if vectors[i] is not None and vectors[i + 1] is not None]

        mreport: dict[str, Any] = {
            "first_frame_similarity": {
                "min": min(sims_valid), "median": median(sims_valid),
                "max": max(sims_valid)},
            "consecutive_similarity": summarize(consec),
            "sim_curve_frame0": sim_curve,
            "by_threshold": {},
        }

        for th in thresholds:
            m = incremental_map(vectors, sim_fn, th)
            m["locality"] = locality_stats(m["assignment"])
            m["revisit"] = revisit_report(m["assignment"], m["n_nodes"])
            # 落盘具体指派太长，只留摘要
            m.pop("assignment", None)
            mreport["by_threshold"][str(th)] = m

        report["methods"][method] = mreport

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"\n[报告已写入 {args.json_out}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
