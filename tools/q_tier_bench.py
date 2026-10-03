# -*- coding: utf-8 -*-
"""建图耗时基准：把录制按真实节奏灌进去，量 ``rasterize()`` 每次多贵、花在哪。

    python -m tools.q_tier_bench [录制名] [--rows]

回答两个问题：
  1. 质量分层（``q_tiers``）让 rasterize 贵了多少？——它加了 4 个累加器，
     ``_grow_acc`` 的搬迁和 ``_sync_acc`` 的 bincount 都是 4→8 倍；
  2. 面板上那个"建图耗时 103 ms"到底花在哪一段？

按真实节奏灌：关键帧每 ``map_min_interval_s``（0.3 s）来一次就栅格化一次，
与 ``OnlineNavigator._mapping_loop`` 的行为一致（dirty 才栅格化）。
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_grid import FREE, OCC, UNK                # noqa: E402
from backend.nav_mapping import KeyframeGridMapper, MapperConfig  # noqa: E402

REC = "20261001_044153"
MAP_MIN_INTERVAL_S = 0.3
STAGES = ("_rotated", "_update_cam_h", "_sync_acc", "_sync_ray", "_ray_veto", "walked", "_kf_base")


def instrument(m: KeyframeGridMapper, acc: dict[str, list[float]]) -> KeyframeGridMapper:
    """给各阶段挂计时（只统计本 mapper 的调用，避免噪声）。"""
    for name in STAGES:
        fn = getattr(m, name)

        def wrap(fn=fn, name=name):
            def inner(*a, **k):
                t0 = time.perf_counter()
                try:
                    return fn(*a, **k)
                finally:
                    acc[name].append((time.perf_counter() - t0) * 1e3)
            return inner
        setattr(m, name, wrap())
    return m


def run(rec: Path, cfg: MapperConfig, rebuild_every_kf: int = 1) -> dict:
    events = [json.loads(x) for x in
              (rec / "events.jsonl").read_text(encoding="utf-8").strip().splitlines() if x.strip()]
    fp = rec / "final_poses.npz"
    pose_of: dict[int, np.ndarray] = {}
    if fp.exists():
        z = np.load(fp)
        pose_of = {int(i): np.asarray(T, np.float64) for i, T in zip(z["ids"], z["T"])}
    cache: dict[int, np.ndarray] = {}

    def kf(k: int):
        if k not in cache:
            cache.clear()
            cache[k] = np.load(rec / "kf" / f"{k:06d}.npz")
        return cache[k]

    acc: dict[str, list[float]] = defaultdict(list)
    m = instrument(KeyframeGridMapper(cfg), acc)
    per_call: list[float] = []
    cells: list[int] = []
    t0 = time.perf_counter()
    n_built = 0
    for e in events:
        if e.get("kind") == "kf":
            k = int(e["k"])
            T = pose_of.get(k)
            if T is None:
                try:
                    T = np.asarray(kf(k)["T_map"], np.float64)
                except (FileNotFoundError, KeyError, OSError):
                    continue
            if e.get("refresh"):
                m.drop_points(k - 1)
            if k in m._pts:
                m.drop_points(k)
            m.add_keyframe(k, kf(k)["pts"].astype(np.float32), T, e.get("osc"))
        elif e.get("kind") == "trail":
            k = int(e["k"])
            if k in pose_of and e.get("T_dr") is not None:
                m.add_trail(k, pose_of[k] @ np.asarray(e["T_dr"], np.float64), e.get("osc"))
        if m._dirty and n_built % rebuild_every_kf == 0:
            t1 = time.perf_counter()
            ng = m.rasterize()
            per_call.append((time.perf_counter() - t1) * 1e3)
            cells.append(int(ng.grid.size))
            n_built += 1
    total = (time.perf_counter() - t0) * 1e3
    return {"per_call": per_call, "stages": acc, "total_ms": total, "n_build": n_built,
            "n_kf": len(m), "last_cells": cells[-1] if cells else 0}


def pct(a: list[float], q: float) -> float:
    return float(np.percentile(a, q)) if a else 0.0


def report(tag: str, r: dict) -> None:
    pc = r["per_call"]
    print(f"\n=== {tag} ===  关键帧 {r['n_kf']} / 栅格化 {r['n_build']} 次 / 末图 {r['last_cells']} 格")
    print(f"  每次 rasterize:  p50 {pct(pc, 50):6.1f} ms   p95 {pct(pc, 95):6.1f} ms   "
          f"最大 {max(pc):6.1f} ms   合计 {sum(pc) / 1000:6.1f} s")
    print(f"  灌数据+栅格化全程: {r['total_ms'] / 1000:.1f} s")
    print("  分阶段（合计 / 每次均值 / 单次最大）:")
    for name, vals in r["stages"].items():
        if not vals:
            continue
        print(f"    {name:16s} {sum(vals) / 1000:7.2f} s  {sum(vals) / len(vals):6.2f} ms  "
              f"{max(vals):7.2f} ms   n={len(vals)}")


def main() -> None:
    rec_name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else REC
    rec = ROOT / "navmesh_recordings" / rec_name
    cases = (
        ("q_tiers=off（改动前）", MapperConfig(q_tiers=False)),
        ("q_tiers=on （默认）", MapperConfig(q_tiers=True)),
        ("q_tiers=on + ray_clear=off", MapperConfig(q_tiers=True, ray_clear=False)),
    )
    print(f"录制：{rec_name}   栅格化节奏：每来一批 dirty 就来一次（面板口径，非固定 0.3 s 轮询）")
    for tag, cfg in cases:
        report(tag, run(rec, cfg))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
