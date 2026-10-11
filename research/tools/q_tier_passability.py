# -*- coding: utf-8 -*-
r"""q_tiers 清障的**通行性后果**：不相交分类 + 单调性核验 + frontier 影响。

承 q_tier_clear_dest.py 与 q_tier_safety_relevance.py。本脚本做三件事：
  1. 把被清格分成**互不相交**的四类（A/B 会重叠，上一版用加法校验失败）；
  2. 核验一条**结构性结论**：q_tiers 只能把 OCC 变成 FREE/UNK，而
     nav_grid.build:248  passable = (free_obs | walked_mask) & ~occ
     ⇒ 去掉 OCC 只会让 passable 增加、dt 增加 ⇒ **可走中心区单调不减**。
     所以"清障会不会封死路"在结构上不可能，只需量"会不会多开门"。
  3. frontier 计数变化（OCC->UNK 会新造 FREE∩UNK 边界 ⇒ 影响**探索行为**，非安全）。
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
        walked = m.walked()
        wm = ng.walked_mask(walked, 0.20)
        ng.build(radius_m=0.25, walked=walked)
        got[tag] = (ng, wm, m)
    ng_off, _, _ = got["off"]
    ng_on, wm, m_on = got["on"]
    if ng_off.grid.shape != ng_on.grid.shape:
        print("%s SHAPE MISMATCH, skip" % name); return
    g_off, g_on = ng_off.grid, ng_on.grid
    cleared = (g_off == OCC) & (g_on != OCC)
    n_cl = int(cleared.sum())
    cw2 = (ng_on.meta.resolution_m * ng_on.meta.world_scale) ** 2
    to_free = cleared & (g_on == FREE)
    in_walk = cleared & wm
    # 互不相交四类
    both = in_walk & to_free
    A = in_walk & ~to_free          # 仅轨迹走廊背书（->UNK）
    B = to_free & ~in_walk          # 仅 ->FREE（轨迹未背书）<-- 唯一无独立证据的开洞
    C = cleared & ~in_walk & ~to_free   # 惰性
    tot = int(A.sum()) + int(B.sum()) + int(both.sum()) + int(C.sum())

    print("=" * 88)
    print("rec %s  range_m=%s ray_beta=%s ray_step=%s" % (name, info.get("range_m"), cfg_rec.ray_beta, cfg_rec.ray_step_m))
    print("=" * 88)
    print("被清格 %d" % n_cl)
    print("  A  仅走廊背书(->UNK 且 walked)   : %6d (%5.2f%%)" % (int(A.sum()), 100.0*A.sum()/max(n_cl,1)))
    print("  AB 走廊+->FREE                  : %6d (%5.2f%%)" % (int(both.sum()), 100.0*both.sum()/max(n_cl,1)))
    print("  B  仅 ->FREE（**无独立背书**）    : %6d (%5.2f%%)  <== 真正的安全敞口" % (int(B.sum()), 100.0*B.sum()/max(n_cl,1)))
    print("  C  惰性(->UNK 且不在走廊)        : %6d (%5.2f%%)  <== 通行性零后果" % (int(C.sum()), 100.0*C.sum()/max(n_cl,1)))
    print("  互斥校验 sum==cleared : %s (%d vs %d)" % (tot == n_cl, tot, n_cl))

    # 单调性：可走中心区只增不减
    lost = int((ng_off.center & ~ng_on.center).sum())
    gained = int((ng_on.center & ~ng_off.center).sum())
    print("  walkable center: off %d -> on %d   (+%d / -%d 格)"
          % (int(ng_off.center.sum()), int(ng_on.center.sum()), gained, lost))
    print("  **单调性核验（结构结论：永不减）**: lost == 0 -> %s" % (lost == 0))

    # frontier（探索行为，非安全）
    for tag, ng in (("off", ng_off), ("on", ng_on)):
        g = ng.grid
        from scipy.ndimage import binary_dilation
        touch = binary_dilation(g == UNK, np.ones((3, 3), bool))
        print("  frontier(FREE∩UNK 边界) %s: %d 格" % (tag, int(((g == FREE) & touch).sum())))
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
