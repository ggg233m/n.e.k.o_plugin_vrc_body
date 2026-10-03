# -*- coding: utf-8 -*-
"""诊断：改动后被降级/删掉的障碍块，各自的票长什么样。

    python -m tools.q_tier_why 20261001_044153 [topN]

回答一个问题：新规则清掉的到底是"纯远场票的量化倾斜孤岛"，还是**真结构**。
判据看每块（baseline 的 8 连通障碍块）的中位票：
    n_o   格内障碍原始点数     o_n  近距(<q_near_m)障碍票
    o_m   中距(q_near~q_mid)障碍票  o_mk 投过中距票的关键帧数
    g_n   判回空地用的地面票(<q_free_m)  远/障 远带(>q_mid)障碍票 / 总障碍票

真结构的指纹：o_n 大，或 o_mk≥2。量化倾斜孤岛的指纹：票几乎全在远带、o_n≈0、o_mk≤1。
实测（044153）：留住的格远带票占比中位 0.13 / 障碍点中位 148；清掉的 1.00 / 15。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import MapperConfig            # noqa: E402
from tools.q_tier_ab import replay                       # noqa: E402

REC = "20261001_044153"
TOP = 12


def to_grid(acc: np.ndarray, mapper, ng) -> np.ndarray:
    """累加器布局 → NavGrid 布局。

    对齐用 mapper._grid_lo（rasterize 实际用的 lx/ly），不在这里重推 lo_i——推错一次整张表
    就静默错位。**栅格是翻转过的**（rasterize 末尾 g[::-1]）：NavGrid 第 0 行是最大 y，
    所以 NavGrid 行 r ↔ 栅格 y 序 gh-1-r ↔ 累加器行 ly + (gh-1-r) - ay。漏掉这个翻转
    不会报错，只会让所有票型统计上下镜像，看着还挺像回事。
    """
    lx, ly = mapper._grid_lo
    ax, ay = mapper._acc_lo
    gh, gw = ng.grid.shape
    out = np.zeros((gh, gw), float)
    ah, aw = acc.shape
    r_lo = max(0, ly + (gh - 1) - (ay + ah - 1))
    r_hi = min(gh - 1, ly + (gh - 1) - ay)
    c_lo = max(0, lx - ax)
    c_hi = min(gw - 1, lx - ax + aw - 1)
    if r_hi < r_lo or c_hi < c_lo:
        return out
    a_top = ly + (gh - 1) - r_lo - ay          # NavGrid 行 r_lo 对应的累加器行（区间内最大）
    out[r_lo:r_hi + 1, c_lo:c_hi + 1] = \
        acc[a_top - (r_hi - r_lo):a_top + 1, lx - ax + c_lo:lx - ax + c_hi + 1][::-1]
    return out


def main() -> None:
    rec_name = sys.argv[1] if len(sys.argv) > 1 else REC
    top = int(sys.argv[2]) if len(sys.argv) > 2 else TOP
    rec = ROOT / "navmesh_recordings" / rec_name

    ng_off, _t, _s, _m = replay(rec, MapperConfig(q_tiers=False))
    ng_on, trail, _s, m = replay(rec, MapperConfig(q_tiers=True))
    base = ng_off.grid == 0
    after = ng_on.grid == 0

    n_o = to_grid(m._acc_o, m, ng_on)
    o_n = to_grid(m._acc_on, m, ng_on)
    o_m = to_grid(m._acc_om, m, ng_on)
    o_mk = to_grid(m._acc_omk, m, ng_on)
    g_n = to_grid(m._acc_gn, m, ng_on)
    n_g = to_grid(m._acc_g, m, ng_on)
    far = np.maximum(n_o - o_n - o_m, 0.0)

    # 自检：对齐对了的话，每个 baseline 障碍格必然 n_o > min_pts-0.5。
    chk = float((n_o[base] <= 2.5).mean())
    print(f"\n=== {rec_name}  改动前障碍格 {int(base.sum())} → 改动后 {int(after.sum())} ===")
    print(f"对齐自检：baseline 障碍格里 n_o<=min_pts 的比例 = {chk:.4f}（应为 0）")

    n, lab, stats, _c = cv2.connectedComponentsWithStats(base.astype(np.uint8), connectivity=8)
    area = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-area)[:top]
    hdr = (f"{'块#':>4} {'格数':>6} {'n_o':>7} {'o_n':>7} {'o_m':>7} {'o_mk':>5} "
           f"{'g_n':>7} {'远/障':>6} {'近/障':>6} {'留存':>6}")
    print(f"\n按格数排前 {top} 的块（数值 = 块内中位票）")
    print(hdr)
    print("-" * 96)
    for r in order:
        cid = int(r) + 1
        msk = lab == cid
        no = n_o[msk]
        keep = float(after[msk].mean())
        far_sh = float(np.median(far[msk] / np.maximum(no, 1.0)))
        near_sh = float(np.median((o_n[msk] + g_n[msk]) / np.maximum(no + n_g[msk], 1.0)))
        print(f"{cid:>4} {int(area[r]):>6} {np.median(no):>7.1f} {np.median(o_n[msk]):>7.1f} "
              f"{np.median(o_m[msk]):>7.1f} {np.median(o_mk[msk]):>5.1f} {np.median(g_n[msk]):>7.1f} "
              f"{far_sh:>6.2f} {near_sh:>6.2f} {keep:>5.0%}")

    # 全局：留存率 vs 票型，看规则是不是在按"证据强度"分层
    kept = base & after
    lost = base & ~after
    print(f"\n全局票型对比（baseline 障碍格 {int(base.sum())}）")
    for tag, msk in (("留存", kept), ("清掉", lost)):
        no = n_o[msk]
        print(f"  {tag}: n_o中位={np.median(no):6.1f}  o_n中位={np.median(o_n[msk]):6.1f}  "
              f"o_m中位={np.median(o_m[msk]):6.1f}  o_mk中位={np.median(o_mk[msk]):4.1f}  "
              f"远带占比中位={np.median(far[msk] / np.maximum(no, 1.0)):.2f}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
