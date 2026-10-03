# -*- coding: utf-8 -*-
"""``_by_quality`` 里"近带故意不带地面压制"这句注释，是活的吗？

    python -m tools.near_band_check [录制名...]

## 疑点

``_by_quality`` 的文档写着：

    近带 <q_near_m：……**故意不带**地面压制条件，
    否则 occ_ground_ratio 会把真细障碍（桌腿、栏杆）一起误杀

但 ``_by_quality`` 拿到的 ``base_occ`` 是 ``rasterize`` 第 900 行算的，那里**已经**带了
地面压制：``(n_o > min_pts) & (n_o >= occ_ground_ratio * n_g)``。
而 ``occ = (base_occ > 0) & (near | mid | solo)``——**near 只能在 base_occ 已经是 1 的
格上加分，不能把被地面压制掉的格救回来。**

所以：近带再多的票，只要该格地面票多，base_occ 就是 0，near 根本轮不到。
**桌腿和栏杆正是这种形状：立面窄、背后的地面看得很清楚。**

## 量什么

    near 成立但 base_occ=0 的格 —— 文档里想救的那批
    其中被地面压制单独杀掉的   —— 真正"死代码"的规模
    它们现在的归宿             —— free / unknown
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import KeyframeGridMapper, MapperConfig    # noqa: E402
from backend.nav_grid import FREE, OCC, UNK                          # noqa: E402
from tools.precision_probe import load                               # noqa: E402

RECS = ("20261001_044153", "20260929_045615")


class Spy(KeyframeGridMapper):
    def _by_quality(self, base_occ, n_g, n_o, o_n, o_m, g_n, o_mk):
        occ, restored, rejected = super()._by_quality(base_occ, n_g, n_o, o_n, o_m, g_n, o_mk)
        self.spy = {k: v.copy() for k, v in
                    (("base_occ", base_occ), ("n_g", n_g), ("n_o", n_o), ("o_n", o_n),
                     ("o_m", o_m), ("occ", occ), ("restored", restored), ("rejected", rejected))}
        return occ, restored, rejected


def main() -> None:
    for name in (sys.argv[1:] or list(RECS)):
        frames = load(ROOT / "navmesh_recordings" / name)
        if not frames:
            print(f"\n=== {name} === 没有关键帧")
            continue
        c = MapperConfig()
        m = Spy(c)
        for i, fr in enumerate(frames):
            m.add_keyframe(i, fr[1], fr[2])
        ng = m.rasterize()
        s = {k: (v[::-1] if v.ndim == 2 else v) for k, v in m.spy.items()}
        g = ng.grid
        hi = c.min_pts - 0.5
        near = s["o_n"] > hi
        bo = s["base_occ"] > 0
        n_o, n_g = s["n_o"], s["n_g"]
        print(f"\n=== {name} ===  {len(frames)} 关键帧   "
              f"occ_ground_ratio={c.occ_ground_ratio}  min_pts={c.min_pts}")
        print(f"  近带（<{c.q_near_m} m）有打的格          {int(near.sum()):7d}")
        print(f"  其中 base_occ 已判障碍                    {int((near & bo).sum()):7d}")
        print(f"  其中 base_occ=0 —— 文档想救、实际救不回  {int((near & ~bo).sum()):7d}")
        dead = near & ~bo & (n_o > hi)          # 票够多，纯粹被地面压制杀掉
        print(f"    └ 其中票数>min_pts、**只**被地面压制杀掉的 {int(dead.sum()):7d}  "
              f"（需 n_o≥{c.occ_ground_ratio}·n_g）")
        if dead.any():
            ngn = n_g[dead]
            print(f"      这批 n_g 中位 {np.median(ngn):.1f}   n_o 中位 {np.median(n_o[dead]):.1f}   "
                  f"门槛中位 {c.occ_ground_ratio * np.median(ngn):.1f}")
            dest = np.where(g[dead] == FREE, "FREE(可走)", np.where(g[dead] == OCC, "OCC", "UNK"))
            for k in ("FREE(可走)", "UNK", "OCC"):
                n = int((dest == k).sum())
                if n:
                    print(f"      归宿 {k:10s} {n:7d} 格 ({n / int(dead.sum()) * 100:5.1f}%)")
        print(f"  参照：全图 OCC {int((g == OCC).sum())}  FREE {int((g == FREE).sum())}  "
              f"UNK {int((g == UNK).sum())}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
