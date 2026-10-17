#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用**已过滤的栅格**量「同一面墙在多会话里被放在哪」的离散度。

为什么不用稠密点云
------------------
``xsess_wall_scatter.py``（点云版）做过一版，但它的**分半基线给 30–39 cm** ——
同一会话、同一套位姿、同一面墙，只看一半帧 vs 另一半帧就差这么多 ⇒
**稠密点云跨视角配准在这个数据上根本不可靠**（与"稠密 ICP 短跨度对照错 1.88 m"
同源）。原始点云约一半是弱纹理系统性误匹配，不能拿来定义"墙在哪"。

本脚本改用 ``offline_fusion`` 产出的**三态栅格**（射线清除 + 中距多帧确认 +
射线式降级），噪声应当低得多。

为什么它自带交叉验证
--------------------
点云版扣掉基线后估计真实差 ≈ 24 cm。本脚本若给出接近的数 ⇒ 两条独立通路
互相印证，结论可信度大幅提升；若显著更小 ⇒ 点云版那 24 cm 里还有残留的方法噪声。

做法
----
1. 每个会话：``offline_fusion.aggregate(rec, "merged", ...)`` → 栅格落在**主 gauge**
   （``ix/iy`` 是世界坐标 ÷ res_m，merged 位姿 ⇒ 各会话同坐标系）。
2. ``dense`` → ``veto_grid`` → ``classify`` → ``labels_f``（0 未知 / 1 空 / 2 障碍）。
3. 取障碍掩码，窗口内 2D 互相关 + 抛物线亚像素，找最优平移。

 **绝不带 ``--write-prior``**：那会覆盖盘上唯一那份先验。本脚本只调函数，不写先验。

用法
----
    python research/tools/xsess_grid_scatter.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import research.tools.offline_fusion as OF          # noqa: E402
import research.tools.xsess_wall_scatter as XS      # noqa: E402  复用互相关
from backend.nav_mapping import MapperConfig        # noqa: E402

REC = ROOT / "navmesh_recordings"
WIN_M = 3.0


def build_grid(rec: str, mode: str = "merged", ray_clear: bool | None = None):
    """返回 (障碍掩码 float, 世界原点 xy 米, labels)。"""
    sid = rec
    world_dir = OF.find_world(sid)
    agg = OF.aggregate(REC / rec, mode, world_dir, sid, ray_clear=ray_clear)
    x0, y0, gband, oband, sband = OF.dense(agg)
    H, W = gband.shape[:2]
    veto = OF.veto_grid(agg, x0, y0, (H, W))
    _occ_b, labels_f, _occ_b3, st = OF.classify(gband, oband, sband, veto)
    res = float(MapperConfig().res_m)
    occ = (labels_f == 2).astype(np.float64)
    return occ, np.array([x0 * res, y0 * res]), labels_f, st, res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rec", nargs="+",
                    default=["20261001_044153", "20261006_001523",
                             "20261010_033237", "20261013_223033"])
    ap.add_argument("--pose", default="merged", choices=("auto", "final", "merged", "rigid"),
                    help="merged = 用已采纳的跨会话位姿 ⇒ 各会话同坐标系")
    ap.add_argument("--win-m", type=float, default=WIN_M)
    ap.add_argument("--min-cells", type=int, default=12,
                    help="窗口内单场最少障碍格数（栅格比点云稀，阈值要低）")
    ap.add_argument("--max-shift", type=float, default=0.60)
    a = ap.parse_args()

    grids, origins, res = {}, {}, None
    print(f"位姿口径：{a.pose}（merged ⇒ 各会话已落在主 gauge 下）")
    for name in a.rec:
        occ, o, lab, st, res = build_grid(name, a.pose)
        grids[name], origins[name] = occ, o
        n_occ = int((lab == 2).sum())
        n_unk = int((lab == 0).sum())
        print(f"  {name}: 栅格 {lab.shape}  障碍 {n_occ}  未知 {n_unk}  "
              f"原点 ({o[0]:.1f}, {o[1]:.1f}) m")

    XS.CELL = res
    W = int(round(a.win_m / res))
    print(f"\n格 {res * 100:.0f} cm，窗口 {a.win_m} m（{W}×{W} 格），"
          f"单场最少 {a.min_cells} 个障碍格\n")
    print(f"{'会话对':<40}{'窗口':>6}{'平移中位 cm':>13}{'p25':>8}{'p75':>8}{'系统偏移 cm':>13}")
    print("-" * 88)

    all_sh = []
    for i in range(len(a.rec)):
        for j in range(i + 1, len(a.rec)):
            na, nb = a.rec[i], a.rec[j]
            S = XS.compare_pair(grids[na], origins[na], grids[nb], origins[nb],
                                W, a.min_cells, a.max_shift)
            if len(S) < 5:
                print(f"{na + ' / ' + nb:<40}{len(S):>6}   （窗口太少）")
                continue
            mag = np.linalg.norm(S, axis=1) * 100
            mx, my = np.median(S[:, 0]) * 100, np.median(S[:, 1]) * 100
            all_sh.append(S)
            print(f"{na + ' / ' + nb:<40}{len(S):>6}{np.median(mag):>13.1f}"
                  f"{np.percentile(mag, 25):>8.1f}{np.percentile(mag, 75):>8.1f}"
                  f"{np.hypot(mx, my):>13.1f}")

    if not all_sh:
        print("\n没有可用的会话对")
        return 1
    A = np.concatenate(all_sh)
    mag = np.linalg.norm(A, axis=1) * 100
    print("\n" + "=" * 88)
    print(f"全体：{len(A)} 个窗口，跨会话墙位置离散度（栅格口径）")
    print(f"  中位 {np.median(mag):.1f} cm   p25 {np.percentile(mag, 25):.1f}   "
          f"p75 {np.percentile(mag, 75):.1f}   p90 {np.percentile(mag, 90):.1f}")
    print(f"  分量：dx 中位 {np.median(A[:, 0]) * 100:+.1f} ± {np.std(A[:, 0]) * 100:.1f}   "
          f"dy 中位 {np.median(A[:, 1]) * 100:+.1f} ± {np.std(A[:, 1]) * 100:.1f} cm")
    print()
    print("交叉验证：")
    print("  · 点云版（扣掉 32.5 cm 基线后）给 ≈ 24 cm")
    print(f"  · 本栅格版给 {np.median(mag):.1f} cm")
    print("  ⇒ 接近则两条独立通路互相印证；本版显著更小说明点云版仍有残留噪声")
    print(f"  · {len(a.rec)} 场平均后 → {np.median(mag) / np.sqrt(len(a.rec)):.1f} cm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
