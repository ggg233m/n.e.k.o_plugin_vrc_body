# -*- coding: utf-8 -*-
r"""把 2–3 m 带从「判 free」改成「诚实降级 UNK」的**代价**实测 —— 不需外部真值。

起因：Docs/虚假障碍清障安全性裁定…（2026-10-11）§三.2 量到分辨力**在 2–3 m 归零**
（复现格 85.2% vs 单半格 85.4%，比值 1.00×），⇒ 那一带的"清"没有证据支撑。
§六.3 提了出路「改成降级 UNK」，但**代价没量**。

本工具量代价。判据不是"清掉几格"，而是**规划器实际还能不能走**：
  1. 三态计数差（FREE 少多少、UNK 多多少）
  2. **可达/可走面积变化**（NavGrid.build 后可走区）
  3. **被清格是否落在可走区内部**——内部格降级会挖洞，边缘格降级只是拔掉毛刺
  4. 走廊 0.5 m 内还剩几个障碍（mapping_gate 同口径，必须并排读）

⚠️ 本工具**不改建图规则**，只在**栅格后处理**上模拟：把"on 时被清掉、且原格到轨迹距离
落在 2–3 m"的格从 FREE 翻回 UNK，其余不动。这是那段改动的**上界代价**
（真改规则时部分格本就不会变 FREE）。
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

FAR_BANDS = {"2-3m": (2.0, 3.0)}


def dist_to_trail(ng, trail):
    """每格到 walked 折线的距离（追踪米）。复用 wallcheck 的稠密化口径。"""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    pts = np.asarray(trail, np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return np.full(ng.grid.shape, np.nan)
    segs = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        n = max(2, int(np.hypot(*(b - a)) / 0.1) + 1)
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        segs.append(a + t * (b - a))
    segs.append(pts[-1:])
    dense = np.concatenate(segs, 0)
    # 格中心（跟踪米）
    m = ng.meta; H, W = ng.grid.shape
    r, c = np.mgrid[0:H, 0:W]
    x = m.origin_xy_m[0] + (c + 0.5) * m.resolution_m
    y = m.origin_xy_m[1] + (H - 1 - r + 0.5) * m.resolution_m
    cent = np.stack([x, y], -1).reshape(-1, 2)
    d, _ = cKDTree(dense).query(cent, k=1)
    return d.reshape(H, W)


def walkable(ng):
    """可走中心区面积（世界 m²）——规划器真正能用的那部分。"""
    cw = ng.meta.resolution_m * ng.meta.world_scale
    try:
        ng.build(radius_m=0.25, walked=None)
    except Exception:
        pass
    c = getattr(ng, "center", None)
    return (float(c.sum()) * cw * cw if c is not None else float("nan"))


def corridor_blockers(ng, trail, radius_m=0.5):
    tm = TG.trail_mask(ng, trail, radius_m)
    return int(((ng.grid == OCC) & tm).sum())


def analyze(name):
    rec = ROOT / "navmesh_recordings" / name
    cfg_rec, info = MG.recorded_config(rec)
    res = {}
    for tag, qt in (("off", False), ("on", True)):
        ng, trail, st, m = MG.replay(rec, dataclasses.replace(cfg_rec, q_tiers=qt))
        res[tag] = (ng, trail, m)
    ng_off, trail_off, _ = res["off"]
    ng_on, trail_on, _ = res["on"]

    assert ng_off.grid.shape == ng_on.grid.shape, (ng_off.grid.shape, ng_on.grid.shape)
    g_off, g_on = ng_off.grid, ng_on.grid
    cleared = (g_off == OCC) & (g_on != OCC)
    dist = dist_to_trail(ng_on, trail_on)

    print("=" * 84)
    print("录制 %s   %s" % (name, info))
    print("=" * 84)
    base = {"FREE": int((g_on == FREE).sum()), "UNK": int((g_on == UNK).sum()),
            "OCC": int((g_on == OCC).sum())}
    print("on 基线三态: FREE=%d UNK=%d OCC=%d" % (base["FREE"], base["UNK"], base["OCC"]))
    print("被清格总数: %d" % int(cleared.sum()))

    # 只在 2-3 m 带的被清格上模拟"降级"
    for band, (lo, hi) in FAR_BANDS.items():
        sel = cleared & (dist >= lo) & (dist < hi)
        # 只处置原本被判成 FREE 的那些（判 UNK 的已经是降级）
        flips = sel & (g_on == FREE)
        n_flip = int(flips.sum())
        g_new = g_on.copy()
        g_new[flips] = UNK
        print()
        print("--- 带 %s：清掉 %d 格，其中现在是 FREE 的 %d 格可降级 ---"
              % (band, int(sel.sum()), n_flip))
        if not n_flip:
            continue
        cw = ng_on.meta.resolution_m * ng_on.meta.world_scale
        print("    降级格面积      : %.2f m²" % (n_flip * cw * cw))
        # 这些格是不是可走区内部？
        try:
            ng_tmp = type(ng_on)(g_new.copy(), ng_on.meta)
            ng_base = type(ng_on)(g_on.copy(), ng_on.meta)
            wa_base, wa_new = walkable(ng_base), walkable(ng_tmp)
            print("    可走区面积      : %.2f -> %.2f m²  (%.2f%%)"
                  % (wa_base, wa_new, (wa_new - wa_base) / wa_base * 100))
        except Exception as exc:
            print("    可走区面积      : 计算失败 %s" % exc)
        cb_base = corridor_blockers(ng_on, trail_on)
        cb_new = corridor_blockers(type(ng_on)(g_new.copy(), ng_on.meta), trail_on)
        print("    走廊 0.5m 障碍  : %d -> %d" % (cb_base, cb_new))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", nargs="*", default=["20261001_044153"])
    a = ap.parse_args()
    for name in a.rec:
        analyze(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
