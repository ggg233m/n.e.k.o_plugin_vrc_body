"""只读验收已有 2.5D 经验地图，不运行点云或真实动作。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(run_dir: str) -> dict:
    run = Path(run_dir).resolve()
    files = {
        name: run / name
        for name in ("run.json", "topo_map.json", "pose_graph.json", "nav_map.json", "world_model.json")
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        return {"mode": "2d_map_acceptance", "status": "blocked", "missing": missing}

    run_json = _load(files["run.json"])
    topo = _load(files["topo_map.json"])
    nav = _load(files["nav_map.json"])
    world = _load(files["world_model.json"])
    meta = nav.get("meta") or {}
    nodes = nav.get("nodes") or []
    route_edges = nav.get("route_edges") or []
    alias_edges = nav.get("alias_edges") or []
    unknown_regions = nav.get("unknown_regions") or []
    node_ids = {int(node["group_id"]) for node in nodes}
    unknown_ids = {int(item["group_id"]) for item in unknown_regions}
    offset = (run_json.get("video_timebase") or {}).get("offset_s")
    topo_offset = topo.get("video_timebase_offset_s")

    checks = {}
    checks["required_artifacts"] = True
    checks["timebase_present"] = offset is not None and topo_offset is not None
    checks["timebase_matches"] = checks["timebase_present"] and abs(float(offset) - float(topo_offset)) < 1e-6
    checks["route_edges_reference_known_nodes"] = all(
        int(edge.get("from")) in node_ids and int(edge.get("to")) in node_ids
        for edge in route_edges
    )
    checks["alias_edges_rejected"] = all(
        edge.get("type") == "alias" and edge.get("traversable") is False
        for edge in alias_edges
    )
    checks["unknown_regions_explicitly_guarded"] = all(
        (int(edge.get("from")) in unknown_ids or int(edge.get("to")) in unknown_ids)
        == bool(edge.get("passes_unknown"))
        for edge in route_edges
    )
    checks["world_self_check"] = all(
        bool(item.get("ok")) for item in (world.get("self_check") or {}).get("checks", [])
    )
    route_sum = sum(float(edge.get("distance_m") or 0.0) for edge in route_edges)
    meta_route_sum = float(meta.get("osc_route_sum_zoh_m") or 0.0)
    checks["route_distance_matches_meta"] = abs(route_sum - meta_route_sum) < 0.01
    checks["all_route_costs_uncertain_are_explicit"] = all(
        bool(edge.get("cost_uncertain")) == (float(edge.get("osc_max_gap_s") or 0.0) > float(meta.get("osc_gap_uncertain_s") or 0.25))
        for edge in route_edges
    )
    checks["map_preview_exists"] = (run / "map_preview.html").exists()

    passed = all(checks.values())
    action_path = run / "action_timeline.jsonl"
    return {
        "mode": "2d_map_acceptance",
        "status": "pass" if passed else "failed",
        "primary_status": "pass" if passed else "failed",
        "run_id": run_json.get("run_id") or run.name,
        "projection": "2.5D",
        "checks": checks,
        "counts": {
            "nodes": len(nodes),
            "n_nav_nodes": int(meta.get("n_nav_nodes") or 0),
            "n_obs_nodes": int(meta.get("n_obs_nodes") or 0),
            "route_edges": len(route_edges),
            "alias_edges": len(alias_edges),
            "unknown_regions": len(unknown_regions),
            "cost_uncertain_edges": sum(bool(edge.get("cost_uncertain")) for edge in route_edges),
        },
        "distance": {
            "route_sum_zoh_m": meta.get("osc_route_sum_zoh_m"),
            "route_sum_stop_m": meta.get("osc_route_sum_stop_m"),
            "spread_m": meta.get("osc_route_spread_m"),
        },
        "timebase": {"offset_s": offset, "topo_offset_s": topo_offset},
        "route_history": "available" if action_path.exists() else "not_available",
        "pointcloud_3d": "optional_not_used_as_gate",
    }


def render(report: dict) -> str:
    c = report.get("counts", {})
    d = report.get("distance", {})
    lines = [
        "# 2.5D 建图主验收报告",
        "",
        "> 本报告只验收 2.5D 经验地图；点云 3D 和真实动作日志不作为本次主验收门槛。",
        "",
        f"- run：`{report.get('run_id')}`",
        f"- 结果：`{report.get('status')}`",
        f"- 时间基准：offset `{report.get('timebase', {}).get('offset_s')}` s，拓扑记录 `{report.get('timebase', {}).get('topo_offset_s')}` s",
        f"- 节点：{c.get('nodes')}（可导航 {c.get('n_nav_nodes')} / 观测 {c.get('n_obs_nodes')}）",
        f"- 路线边：{c.get('route_edges')}；alias：{c.get('alias_edges')}；未知区：{c.get('unknown_regions')}",
        f"- 路线距离：ZOH `{d.get('route_sum_zoh_m')}` m；stop 口径 `{d.get('route_sum_stop_m')}` m；差值 `{d.get('spread_m')}` m",
        f"- 成本不确定边：{c.get('cost_uncertain_edges')} / {c.get('route_edges')}",
        "",
        "## 检查结果",
        "",
    ]
    for name, value in (report.get("checks") or {}).items():
        lines.append(f"- {'通过' if value else '失败'}：`{name}`")
    lines += [
        "",
        "## 边界",
        "",
        f"- 路线历史：`{report.get('route_history')}`；没有 `action_timeline` 时不确认真实动作和地点身份。",
        "- 点云 3D：只作为可选近似预览，不作为 2.5D 导航或碰撞门槛。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="验收已有 2.5D 经验地图")
    ap.add_argument("--run", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--md-out", required=True)
    args = ap.parse_args()
    report = check(args.run)
    Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.md_out).write_text(render(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "json_out": args.json_out, "md_out": args.md_out}, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
