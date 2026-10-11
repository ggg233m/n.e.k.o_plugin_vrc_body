# -*- coding: utf-8 -*-
r"""q_tiers 清障的**去向**与**可走面代价** —— 在线 mapper 本体，不需外部真值。

起因：Docs/虚假障碍清障安全性裁定…（2026-10-11）把"清掉的是不是真墙"当作未决问题，
但**从没量过被清格去了哪**。三态里 OCC 的下游只有两个去处：FREE 或 UNK，而两者
对规划器的含义**完全相反**：

  * nav_grid.NavGrid.build:248  passable = (free_obs | walked_mask) & ~occ
    => **UNK 不可走**（free_obs 只认 FREE）。所以 OCC->UNK 对规划器等于"仍然过不去"，
    只是从"已知障碍"变成"未知"；**OCC->FREE 才是真正的开洞**。
  * 于是"清障安不安全"分成两个可分开量的问题：
      (a) **碰撞安全**：清掉的格有没有变成可走的？（OCC->FREE 的比例）
      (b) **覆盖代价**：可走区面积少了多少？

判据是 NavGrid.build 的 walkable_center_m2（与 mapping_gate 同源），不是格数。
"""
from __future__ import annotations
import argparse, dataclasses, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research" / "tools"))
from backend.nav_grid import FREE, OCC, UNK  # noqa: E402
import mapping_gate as MG                       # noqa: E402
import trail_geom as TG                         # noqa: E402

BANDS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]


def dist_to_trail(ng, trail, step_m=0.1):
    """每格到 walked 折线的距离（追踪米）。折线按 step_m 稠密化后建 KD 树。"""
    from scipy.spatial import cKDTree
    p = np.asarray(trail, np.float64).reshape(-1, 2)
    if len(p) < 2:
        return np.full(ng.grid.shape, np.nan)
    segs = []
    for i in range(len(p) - 1):
        a, b = p[i], p[i + 1]
        n = max(2, int(np.hypot(*(b - a)) / step_m) + 1)
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        segs.append(a + t * (b - a))
    segs.append(p[-1:])
    dense = np.concatenate(segs, 0)
    m = ng.meta; H, W = ng.grid.shape
    rr, cc = np.mgrid[0:H, 0:W]
    x = m.origin_xy_m[0] + (cc + 0.5) * m.resolution_m
    y = m.origin_xy_m[1] + (H - 1 - rr + 0.5) * m.resolution_m
    cent = np.stack([x, y], -1).reshape(-1, 2)
    d, _ = cKDTree(dense).query(cent, k=1)
    return d.reshape(H, W)


def walk_m2(ng):
    ng.build(radius_m=0.25)
    return float(ng.center.sum()) * (ng.meta.resolution_m * ng.meta.world_scale) ** 2


def analyze(name):
    rec = ROOT / "navmesh_recordings" / name
    cfg_rec, info = MG.recorded_config(rec)
    got = {}
    for tag, qt in (("off", False), ("on", True)):
        ng, trail, st, m = MG.replay(rec, dataclasses.replace(cfg_rec, q_tiers=qt))
        got[tag] = (ng, trail)
    ng_off, tr = got["off"]
    ng_on, _ = got["on"]
    if ng_off.grid.shape != ng_on.grid.shape:
        print("%s  SHAPE MISMATCH %s %s (skip)" % (name, ng_off.grid.shape, ng_on.grid.shape))
        return
    g_off, g_on = ng_off.grid, ng_on.grid
    cleared = (g_off == OCC) & (g_on != OCC)
    n_cl = int(cleared.sum())
    d = dist_to_trail(ng_on, tr)
    cw2 = (ng_on.meta.resolution_m * ng_on.meta.world_scale) ** 2

    print("=" * 88)
    print("rec %s   range_m=%s ray_beta=%s ray_step=%s  kf=%s" % (
        name, info.get("range_m"), cfg_rec.ray_beta, cfg_rec.ray_step_m, info.get("recorded_keys")))
    print("=" * 88)
    print("cleared cells total %d" % n_cl)
    nf = int((cleared & (g_on == FREE)).sum())
    nu = int((cleared & (g_on == UNK)).sum())
    print("  dest: ->FREE %d (%.2f%%)   ->UNK %d (%.2f%%)"
          % (nf, 100.0 * nf / max(n_cl, 1), nu, 100.0 * nu / max(n_cl, 1)))
    print("  NOTE only ->FREE is a real hole (UNK is not walkable, see nav_grid.build:248)")

    wa_off, wa_on = walk_m2(ng_off), walk_m2(ng_on)
    print("  walkable center: off %.2f m2 -> on %.2f m2  (%+.2f m2, %+.3f%%)"
          % (wa_off, wa_on, wa_on - wa_off, (wa_on - wa_off) / wa_off * 100))

    cb_off = int(((ng_off.grid == OCC) & TG.trail_mask(ng_off, tr, 0.5)).sum())
    cb_on = int(((ng_on.grid == OCC) & TG.trail_mask(ng_on, tr, 0.5)).sum())
    print("  corridor 0.5m obstacles: %d -> %d" % (cb_off, cb_on))

    print()
    print("  %-9s %8s %8s %8s %8s %9s" % ("band", "cleared", "->FREE", "->UNK", "FRE%", "area m2"))
    for lo, hi in BANDS:
        sel = cleared & (d >= lo) & (d < hi)
        n = int(sel.sum())
        if not n:
            continue
        f = int((sel & (g_on == FREE)).sum())
        u = int((sel & (g_on == UNK)).sum())
        print("  %-9s %8d %8d %8d %8.2f %9.2f"
              % ("%g-%gm" % (lo, hi), n, f, u, 100.0 * f / n, n * cw2))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", nargs="*", default=None)
    a = ap.parse_args()
    names = a.rec or ["20261001_044153", "20260929_045615", "20261010_033237"]
    for n in names:
        if (ROOT / "navmesh_recordings" / n).is_dir():
            analyze(n)
        else:
            print("%s not found, skip" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
