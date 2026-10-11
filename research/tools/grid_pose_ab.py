# -*- coding: utf-8 -*-
r"""同一录制、换位姿源重建栅格，看拖影到底动不动导航地图。

位姿源：
  dr     kf/*.npz 的 T_dr（不含回环）
  loop   final_poses.npz（现行在线：DR + nav_loop 回环）
  <file> <rec>/<file>.npz（ids,T），如 kf_visual_odom.py 的产出
  dr+N   T_dr + 每帧独立 xy 高斯扰动 σ=N 米（敏感度对照：人为制造拖影）

建图走在线 KeyframeGridMapper 本体（mapping_gate.replay 同序），指标沿用 mapping_gate.measure。

    .venv/Scripts/python.exe research/tools/grid_pose_ab.py 20261001_044153 --src dr loop vo_g6 dr+0.1 dr+0.2
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapping_gate as G                                       # noqa: E402
from backend.nav_mapping import KeyframeGridMapper            # noqa: E402
from backend.nav_grid import OCC, FREE                         # noqa: E402


def poses_for(rec: Path, src: str) -> dict[int, np.ndarray]:
    ids = sorted(int(p.stem) for p in (rec / "kf").glob("*.npz"))
    if src == "loop":
        z = np.load(rec / "final_poses.npz")
        return {int(i): np.asarray(T, float) for i, T in zip(z["ids"], z["T"])}
    if src.startswith("dr"):
        P = {k: np.asarray(np.load(rec / "kf" / f"{k:06d}.npz")["T_dr"], float) for k in ids}
        if "+" in src:
            sig = float(src.split("+")[1])
            rng = np.random.default_rng(0)
            for k in ids:
                P[k] = P[k].copy()
                P[k][:2, 3] += rng.normal(0, sig, 2)
        return P
    z = np.load(rec / f"{src}.npz")
    return {int(i): np.asarray(T, float) for i, T in zip(z["ids"], z["T"])}


def replay(rec: Path, cfg, pose_of: dict[int, np.ndarray]):
    """mapping_gate.replay 的逐行拷贝，只把位姿源换成 pose_of（没有的帧跳过）。"""
    events = G.load_events(rec)
    m = KeyframeGridMapper(cfg)
    trail = []
    for e in events:
        kind = e.get("kind")
        if kind not in ("kf", "trail"):
            continue
        k = int(e["k"])
        T = pose_of.get(k)
        if T is None:
            continue
        if kind == "kf":
            pts = np.load(rec / "kf" / f"{k:06d}.npz")["pts"].astype(np.float32)
            if e.get("refresh"):
                m.drop_points(k - 1)
            if k in m._pts:
                m.drop_points(k)
            m.add_keyframe(k, pts, T, e.get("osc"), t_s=e.get("t"))
            trail.append(T[:2, 3])
        else:
            m.add_trail(k, T @ np.asarray(e["T_dr"], np.float64), e.get("osc"))
    return m.rasterize(), trail, m


def one(args):
    name, src, ref_trail = args
    rec = G.ROOT / "navmesh_recordings" / name
    cfg, _ = G.recorded_config(rec)
    ng, trail, m = replay(rec, cfg, poses_for(rec, src))
    # 走廊/环 一律用参照轨迹（loop），否则换位姿源连"走廊在哪"都变了，不可比
    t = trail                # 自己的轨迹：图与轨迹同一坐标系（各位姿源全局偏移不同，跨源用参照轨迹会错位）
    g = ng.grid
    occ, free = g == OCC, g == FREE
    b = ng.build(radius_m=0.25, walked=m.walked())
    sh = G.occ_shape(occ, ng.meta.resolution_m)
    return name, src, {
        "occ": int(occ.sum()), "free": int(free.sum()),
        "walk_m2": b.get("walkable_m2", b.get("walkable_area_m2")),
        "island": b.get("main_island_pct", b.get("largest_island_pct")),
        "corr": int((G._trail_mask(ng, t, 0.5) & occ).sum()),
        "thick": sh["thickness_p50_m"], "bnd": sh["boundary_frac_pct"], "comp": sh["components"],
        "build_keys": sorted(b),
    }, trail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="+")
    ap.add_argument("--src", nargs="+", default=["dr", "loop"])
    a = ap.parse_args()
    # 先跑 loop 拿参照轨迹
    ref = {}
    with ProcessPoolExecutor() as ex:
        for name, src, r, trail in ex.map(one, [(n, "loop", None) for n in a.recs]):
            ref[name] = (r, trail)
        jobs = [(n, s, ref[n][1]) for n in a.recs for s in a.src if s != "loop"]
        res = {(n, s): r for n, s, r, _ in ex.map(one, jobs)}
    for n in a.recs:
        res[(n, "loop")] = ref[n][0]
    print("build keys:", res[(a.recs[0], "loop")]["build_keys"])
    print(f"{'录制':16s} {'位姿源':10s} {'障碍格':>7s} {'自由格':>7s} {'可走m²':>7s} {'主岛%':>6s} {'走廊障碍':>7s} {'墙厚m':>6s} {'边界%':>6s} {'连通块':>6s}")
    for n in a.recs:
        for s in ["loop"] + [x for x in a.src if x != "loop"]:
            r = res[(n, s)]
            print(f"{n:16s} {s:10s} {r['occ']:7d} {r['free']:7d} {str(r['walk_m2']):>7s} {str(r['island']):>6s} "
                  f"{r['corr']:7d} {r['thick']:6.2f} {r['bnd']:6.1f} {r['comp']:6d}")


if __name__ == "__main__":
    main()
