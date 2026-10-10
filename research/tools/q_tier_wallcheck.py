# -*- coding: utf-8 -*-
r"""q_tiers 清障的「有没有删掉真墙」独立裁定 —— 在线 mapper 本体。

与票型正交的两条物理证据：
 1. 轨迹富集（位姿+OSC，与双目分类无关）
 2. 障碍带高度剖面（mapper 自己的 _acc_ob_h/_acc_ob_h2 可加矩）

⚠️ 必须先分层：清掉/保住两组在**到轨迹距离**上差得很大，而距离是主导混杂
（深度噪声 ∝ z²，规则本身也按距离分层）。不分层就是拿两组不同条件的格互比。
本工具按距离带分层，并在带内并排读两组。
"""
from __future__ import annotations
import dataclasses, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research" / "tools"))
from backend.nav_grid import OCC  # noqa: E402
import mapping_gate as MG  # noqa: E402
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None

BANDS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]


def dense_trail(trail, step_m=0.1):
    p = np.asarray(trail, np.float64).reshape(-1, 2)
    if len(p) < 2:
        return p
    segs = []
    for i in range(len(p) - 1):
        a, b = p[i], p[i + 1]
        n = max(2, int(np.hypot(*(b - a)) / step_m) + 1)
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        segs.append(a + t * (b - a))
    segs.append(p[-1:])
    return np.concatenate(segs, 0)


def centers_tracking(ng):
    m = ng.meta; H, W = ng.grid.shape
    r, c = np.mgrid[0:H, 0:W]
    x = m.origin_xy_m[0] + (c + 0.5) * m.resolution_m
    y = m.origin_xy_m[1] + (H - 1 - r + 0.5) * m.resolution_m
    return np.stack([x, y], -1).reshape(-1, 2)


def summarize(dist, mean_h, sd_h, n_o, mask):
    k = mask.reshape(-1)
    n = int(k.sum())
    if n == 0:
        return None
    dd, mh, sd, no = dist.reshape(-1)[k], mean_h.reshape(-1)[k], sd_h.reshape(-1)[k], n_o.reshape(-1)[k]
    fin = np.isfinite(mh)
    return {
        "n": n, "n_o_p50": float(np.median(no)),
        "d_p50": float(np.median(dd)),
        "h_p50": float(np.nanmedian(mh)) if fin.any() else float("nan"),
        "sd_p50": float(np.nanmedian(sd)) if fin.any() else float("nan"),
        "near_line": float(np.nanmean((mh >= 0.28) & (mh <= 0.50)) * 100) if fin.any() else float("nan"),
        "n_pts_ge8": float(np.mean(no >= 8) * 100),
    }


def analyze(name):
    rec = ROOT / "navmesh_recordings" / name
    cfg_rec, info = MG.recorded_config(rec)
    got = {}
    for tag, qt in (("off", False), ("on", True)):
        ng, trail, st, m = MG.replay(rec, dataclasses.replace(cfg_rec, q_tiers=qt))
        got[tag] = (ng, trail, m)
    ng_off, trail_off, m_off = got["off"]
    ng_on = got["on"][0]

    occ_off = ng_off.grid == OCC
    occ_on = ng_on.grid == OCC
    cleared, retained = occ_off & ~occ_on, occ_off & occ_on

    dist = np.full(ng_off.grid.shape, np.nan)
    if cKDTree is not None and len(trail_off) >= 2:
        d, _ = cKDTree(dense_trail(trail_off)).query(centers_tracking(ng_off))
        dist = d.reshape(ng_off.grid.shape)

    n_o = m_off._cut_acc(m_off._acc_o).astype(np.float64)
    sh = m_off._cut_acc(m_off._acc_ob_h).astype(np.float64)
    sh2 = m_off._cut_acc(m_off._acc_ob_h2).astype(np.float64)
    nzs = n_o > 0.5
    mean_h = np.where(nzs, sh / np.maximum(n_o, 1e-9), np.nan)
    var = np.where(nzs, sh2 / np.maximum(n_o, 1e-9) - np.nan_to_num(mean_h) ** 2, np.nan)
    sd_h = np.sqrt(np.maximum(var, 0.0))

    print("=" * 96)
    print(f"{name}  occ off={int(occ_off.sum())} -> on={int(occ_on.sum())}  range_m={info.get('range_m')}")
    print("\n[总体]")
    print(f"  {'集合':<10}{'格数':>7}{'n_o中位':>9}{'到轨迹p50':>11}{'高p50':>8}{'sd_p50':>8}{'贴线%':>8}{'n_o>=8%':>9}")
    for tag, mk in (("清掉", cleared), ("保住", retained)):
        s = summarize(dist, mean_h, sd_h, n_o, mk)
        if s:
            print(f"  {tag:<10}{s['n']:>7}{s['n_o_p50']:>9.0f}{s['d_p50']:>11.2f}{s['h_p50']:>8.3f}"
                  f"{s['sd_p50']:>8.3f}{s['near_line']:>8.1f}{s['n_pts_ge8']:>9.1f}")
    print("\n[按到轨迹距离分层] —— 层内才是可比的两组")
    print(f"  {'距离带':<12}{'清掉n':>8}{'保住n':>8} | {'清掉 高/sd':>16}{'保住 高/sd':>16} | {'清掉n_o':>8}{'保住n_o':>8}")
    for lo, hi in BANDS:
        band = (dist >= lo) & (dist < hi)
        a = summarize(dist, mean_h, sd_h, n_o, cleared & band)
        b = summarize(dist, mean_h, sd_h, n_o, retained & band)
        lbl = f"{lo:g}-{hi:g}m" if hi < 1e8 else f">{lo:g}m"
        if not a and not b:
            continue
        fa = f"{a['h_p50']:.2f}/{a['sd_p50']:.2f}" if a else "-"
        fb = f"{b['h_p50']:.2f}/{b['sd_p50']:.2f}" if b else "-"
        print(f"  {lbl:<12}{(a['n'] if a else 0):>8}{(b['n'] if b else 0):>8} | {fa:>16}{fb:>16} | "
              f"{(a['n_o_p50'] if a else float('nan')):>8.0f}{(b['n_o_p50'] if b else float('nan')):>8.0f}")
    print("  （高=障碍带内点权均值高度(离地,m)，判定线0.30；sd=格内高度标准差）")
    return 0


def main(recs):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for n in recs:
        try:
            analyze(n)
        except Exception as e:
            print(f"[skip] {n}: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["20261001_044153", "20260929_045615"]))
