# -*- coding: utf-8 -*-
r"""q_tiers 清障的**安全相关性** —— 只有一部分被清格真的改变通行性。

承 q_tier_clear_dest.py：95–99.8% 的被清格去向是 **OCC->UNK**，而 UNK 不可走
（nav_grid.build:248  passable = (free_obs | walked_mask) & ~occ，free_obs 只认 FREE）。

但有一条例外必须量：**walked_mask 内的格**（身体真的走过去的那条 0.20 m 走廊）
即使判 UNK 也算 passable。所以"安全相关"的被清格 =

    被清(OCC->非OCC)  AND  walked_mask(0.20 m)

对这一集合，有一条**与双目分类完全独立**的证据：身体物理上走过那里。
真墙不可能被身体穿过 ⇒ 这一集合的清除由独立链（OSC+HMD 轨迹）**直接背书**。
反过来，walked_mask 之外的被清格一律停在 UNK，对通行性**零后果**。

本工具把被清格按"是否进入通行集"分三类，并给出各自规模。判据是通行性，不是格数。
"""
from __future__ import annotations
import argparse, dataclasses, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research" / "tools"))
from backend.nav_grid import FREE, OCC, UNK  # noqa: E402
import mapping_gate as MG                       # noqa: E402


def analyze(name):
    rec = ROOT / "navmesh_recordings" / name
    cfg_rec, info = MG.recorded_config(rec)
    got = {}
    for tag, qt in (("off", False), ("on", True)):
        ng, trail, st, m = MG.replay(rec, dataclasses.replace(cfg_rec, q_tiers=qt))
        walked = m.walked()                      # 世界米折线，与在线 NavSession.compute 同一调用
        wm = ng.walked_mask(walked, 0.20)        # 身体真的走过的那条走廊
        ng.build(radius_m=0.25, walked=walked)   # 与在线同一参数，拿到真实 center
        got[tag] = (ng, trail, wm)
    ng_off, tr, wm = got["off"]
    ng_on = got["on"][0]
    if ng_off.grid.shape != ng_on.grid.shape:
        print("%s SHAPE MISMATCH, skip" % name)
        return
    g_off, g_on = ng_off.grid, ng_on.grid
    cleared = (g_off == OCC) & (g_on != OCC)
    n_cl = int(cleared.sum())
    cw2 = (ng_on.meta.resolution_m * ng_on.meta.world_scale) ** 2

    # 三类：进了通行集(walked 内)、真开洞(->FREE)、其余(->UNK 且不在 walked)
    in_walk = cleared & wm
    to_free = cleared & (g_on == FREE)
    inert   = cleared & ~wm & ~to_free

    print("=" * 86)
    print("rec %s  range_m=%s ray_beta=%s ray_step=%s" % (name, info.get("range_m"), cfg_rec.ray_beta, cfg_rec.ray_step_m))
    print("=" * 86)
    print("被清格 %d" % n_cl)
    print("  A. 进入通行集（被清 且 walked_mask 0.20m） : %6d  (%.2f%%)   <-- 安全相关，由轨迹链背书"
          % (int(in_walk.sum()), 100.0 * in_walk.sum() / max(n_cl, 1)))
    print("  B. 真开洞（->FREE，UNK 之外的通行变化）     : %6d  (%.2f%%)   <-- 唯一改变可走几何的部分"
          % (int(to_free.sum()), 100.0 * to_free.sum() / max(n_cl, 1)))
    print("  C. 惰性（->UNK 且不在 walked）            : %6d  (%.2f%%)   <-- 对通行性零后果"
          % (int(inert.sum()), 100.0 * inert.sum() / max(n_cl, 1)))
    print("  校验 A+B+C == 被清? %s (%d)" % (int(in_walk.sum()) + int(to_free.sum()) + int(inert.sum()) == n_cl,
                                              int(in_walk.sum()) + int(to_free.sum()) + int(inert.sum())))

    # 通行集变化
    w_off = int(ng_off.center.sum()); w_on = int(ng_on.center.sum())
    print("  walkable center 格数: off %d -> on %d  (%+d 格, %+.2f m2)" % (w_off, w_on, w_on - w_off, (w_on - w_off) * cw2))
    gained = ng_on.center & ~ng_off.center
    lost = ng_off.center & ~ng_on.center
    print("    新增可走 %d 格 / 失去可走 %d 格" % (int(gained.sum()), int(lost.sum())))
    if int(gained.sum()):
        # 新增可走格里，有多少落在"被清格"的 1 格邻域内（即由清障直接放开的）
        from scipy.ndimage import binary_dilation
        near_cl = binary_dilation(cleared, np.ones((3, 3), bool))
        print("    其中紧邻被清格(3x3)的: %d 格 (%.1f%%)" % (int((gained & near_cl).sum()),
              100.0 * int((gained & near_cl).sum()) / int(gained.sum())))
        print("    其中紧邻 ->FREE 格的    : %d 格" % int((gained & binary_dilation(to_free, np.ones((3,3),bool))).sum()))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", nargs="*", default=None)
    a = ap.parse_args()
    for n in (a.rec or ["20261001_044153", "20260929_045615", "20261010_033237"]):
        if (ROOT / "navmesh_recordings" / n).is_dir():
            analyze(n)
        else:
            print("%s not found" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
