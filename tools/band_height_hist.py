# -*- coding: utf-8 -*-
"""带内高度分布：能分出"楼板/天花板"和"墙/杂物"吗？

    python -m tools.band_height_hist [录制名]

## 问题

`tools/band_report.py` 只给出"每个带每格有多少点"。但点数区分不了东西：楼板、天花板、
横梁、栏杆、桥面全都会落进同一条带。要做多层行走/净空，必须先能把**水平面**从
**竖直杂物**里分出来。

## 判据

对每个格取带内点的高度直方图，算两个量：

    peak_h   众数高度（点最密的那个 5 cm 高度格）
    conc     落在 peak_h ±10 cm 内的点占比 —— 平面高，杂物低

然后看 **peak_h 的横向连贯性**：相邻格的众数高度差多大。
  * 一张真实平面（天花��/楼板）⇒ 邻格差很小，且 peak_h 本身在大片区域里几乎不变；
  * 竖直杂物/斜面/噪声 ⇒ 邻格差和整图高差同量级，即"随机"。

连贯性用**与随机打乱的对照**比：把 peak_h 在格间随机置换后重算邻格差，
真实平面的邻格差应当**显著小于**随机对照。

## 输出

每条带：参与统计的格数、conc 的中位/p90、邻格差中位、随机对照的邻格差、
以及"连贯的格占比"（邻格差 < 阈值 的比例）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import KeyframeGridMapper, MapperConfig, _ground_correction, _voxelize  # noqa: E402
from tools.precision_probe import load                            # noqa: E402

REC = "20261001_044153"
HBIN = 0.05           # 高度直方图分辨率（追踪米）
H_LO, H_HI = -4.0, 12.0    # 下界放到 −4 m：地面以下若真有下层结构，别被直方图边缘截断
CELL_M = 0.3          # 统计用的粗格（0.1 m 太细，噪声占比高）
MIN_PTS = 20          # 每格至少这么多点才参与统计
COH_TOL = 0.25        # 邻格众数高度差小于它算"连贯"
BANDS = ("below", "lo", "mid", "hi")


def hist(frames, m: KeyframeGridMapper, lo_i: int, hi_i: int) -> tuple[np.ndarray, np.ndarray]:
    """带内点的 (高度格 × 粗格) 联合直方图 + 每格最近观测距离。"""
    c = m.cfg
    nh = int(round((H_HI - H_LO) / HBIN))
    ny = nx = 0
    base = (0, 0)
    parts: list[np.ndarray] = []
    idx: list[np.ndarray] = []
    cnt: list[np.ndarray] = []
    dist: list[np.ndarray] = []
    for i, fr in enumerate(frames):
        m.add_keyframe(i, fr[1], fr[2])
        m._rotated(i)
        rxy, rel, cn, _R = m._rot[i]
        if rxy is None or not len(rxy):
            continue
        h = rel + np.float32(m.cam_h)
        h = h - np.float32(_ground_correction(h, cn, rxy, c.ground_offset_max_m,
                                              c.ground_offset_min_range_m,
                                              c.ground_plane_max_deg, c.ground_plane_band_m))
        t = m._pose[i][:2, 3]
        xy = rxy.astype(np.float64) + t
        gx = np.floor(xy[:, 0] / CELL_M).astype(np.int64)
        gy = np.floor(xy[:, 1] / CELL_M).astype(np.int64)
        if not idx:
            ny, nx = int(gy.max() - gy.min()) + 1, int(gx.max() - gx.min()) + 1
            base = (int(gy.min()), int(gx.min()))
        hb = np.floor((h - H_LO) / HBIN).astype(np.int64)
        ok = ((hb >= 0) & (hb < nh) & (gy >= base[0]) & (gy < base[0] + ny)
              & (gx >= base[1]) & (gx < base[1] + nx)
              & (h <= -c.ground_tol_m if lo_i == 0 else h >= c.obst_top_m)
              & (h < c.obst_top_m if lo_i == 0 else h > c.obst_top_m))
        if lo_i == 1:
            ok &= h <= c.hi_band_m
        elif lo_i == 2:
            ok &= (h > c.hi_band_m) & (h <= c.hi_band_top_m)
        elif lo_i == 3:
            ok &= h > c.hi_band_top_m
        if not ok.any():
            continue
        parts.append(hb[ok])
        idx.append(((gy[ok] - base[0]) * nx + (gx[ok] - base[1])))
        cnt.append(cn[ok].astype(np.float64))
        dist.append(np.hypot(rxy[ok, 0], rxy[ok, 1]).astype(np.float64))
    if not parts:
        return np.zeros((nh, 0, 0)), np.zeros((0, 0))
    hb = np.concatenate(parts)
    cell = np.concatenate(idx)
    w = np.concatenate(cnt)
    H = np.bincount(hb * (ny * nx) + cell, weights=w, minlength=nh * ny * nx)
    dmin = np.full(ny * nx, np.inf)
    np.minimum.at(dmin, cell, np.concatenate(dist))
    return H.reshape(nh, ny, nx), dmin.reshape(ny, nx)


def analyse(H: np.ndarray, label: str, near_mask: np.ndarray | None = None) -> None:
    tot = H.sum(axis=0)
    valid = tot >= MIN_PTS
    if valid.sum() < 50:
        print(f"  {label:6s} 参与统计的格 {int(valid.sum()):6d}（太少，跳过）")
        return
    mode = H.argmax(axis=0)
    peak = H.max(axis=0)
    clipped = int((H[0].sum() + H[-1].sum()))         # 撞上下界的点数（截断要能看见）
    # 带内整体高度分布（对格求和）：这是"带内高度分布"本身，比 mode/conc 直观
    prof = H.sum(axis=(1, 2))
    edges = np.arange(H_LO, H_HI + HBIN, 0.5)
    print(f"  ── {label} 带内高度分布（0.5 m 桶，% 为占该带总点）"
          f"{'   [撞直方图上下界: %d 点]' % clipped if clipped else ''}")
    for a in edges:
        m0 = (np.arange(len(prof)) * HBIN + H_LO >= a) & (np.arange(len(prof)) * HBIN + H_LO < a + 0.5)
        n = float(prof[m0].sum())
        if n <= 0:
            continue
        pct = 100.0 * n / max(float(prof.sum()), 1.0)
        if pct < 0.5:
            continue
        print(f"     {a:+5.1f} .. {a + 0.5:+5.1f} m  {pct:5.1f}%  " + "█" * int(round(pct / 2)))
    # conc：众数 ±10 cm（±2 个高度格）内的点占比。计数非负，"并集" = 逐格取 max。
    near = H.copy()
    for d in (-2, -1, 1, 2):
        near = np.maximum(near, np.roll(H, -d, axis=0))
    # 每格只取"自己众数那一行"上的窗口和 ⇒ 形状回到 (ny, nx)
    yy, xx = np.indices(tot.shape)
    conc = np.where(tot > 0, near[mode, yy, xx] / np.maximum(tot, 1), 0.0)
    conc_v = conc[valid]
    shape = valid.shape
    pk = np.where(valid, H_LO + (mode + 0.5) * HBIN, np.nan)
    d = []
    for ax in (0, 1):
        a, b = pk, np.roll(pk, 1, axis=ax)
        m = np.isfinite(a) & np.isfinite(b)
        if m.any():
            d.append(np.abs(a[m] - b[m]))
    dif = np.concatenate(d) if d else np.array([np.nan])
    rng = float(np.nanmax(pk) - np.nanmin(pk))
    # 随机对照：把众数高度在格间随机置换，邻格差应当退化成整图高差量级
    rs = np.random.default_rng(0)
    flat = pk[valid]
    shuf = np.full(valid.shape, np.nan)
    shuf[valid] = rs.permutation(flat)      # 只打乱有效格的值，NaN 掩码保持不变
    dr = []
    for ax in (0, 1):
        a, b = shuf, np.roll(shuf, 1, axis=ax)
        m = np.isfinite(a) & np.isfinite(b)
        if m.any():
            dr.append(np.abs(a[m] - b[m]))
    difr = np.concatenate(dr) if dr else np.array([np.nan])
    print(f"  {label:6s} 格 {int(valid.sum()):6d}  conc 中位 {np.median(conc_v):.2f} "
          f"p90 {np.percentile(conc_v, 90):.2f}   peak_h {np.nanmin(pk):.1f}~{np.nanmax(pk):.1f} m")
    print(f"         邻格众数高差 中位 {np.median(dif):.3f} m   随机对照 {np.median(difr):.3f} m"
          f"   连贯占比(<{COH_TOL} m) {100.0 * float((dif < COH_TOL).mean()):.1f}%")
    # 对照：conc 低到底是"面真的厚"还是"深度噪声"。近距看过的格 δz 小，conc 应当明显更高。
    if near_mask is not None:
        for tag, sel in (("最近观测<2m", valid & near_mask), ("最近观测>3m", valid & ~near_mask)):
            if sel.sum() < 30:
                continue
            print(f"         {tag}: 格 {int(sel.sum()):5d}  conc 中位 {np.median(conc[sel]):.2f}"
                  f"  peak_h 中位 {np.median(pk[sel]):.2f} m")


def main() -> None:
    rec_name = sys.argv[1] if len(sys.argv) > 1 else REC
    rec = ROOT / "navmesh_recordings" / rec_name
    frames = load(rec)
    if not frames:
        print(f"{rec_name}: 无关键帧")
        return
    m = KeyframeGridMapper(MapperConfig())
    for i, fr in enumerate(frames):
        m.add_keyframe(i, fr[1], fr[2])
    m.rasterize()                      # 先定下 cam_h
    print(f"录制 {rec_name}  {len(frames)} 关键帧  cam_h={m.cam_h:.3f}  "
          f"统计格 {CELL_M} m  高度格 {HBIN} m")
    print(f"  判读：邻格众数高差 << 随机对照 ⇒ 这一带里有**连贯的水平面**（楼板/天花板）\n")
    for li, name in enumerate(BANDS):
        H, dmin = hist(frames, KeyframeGridMapper(MapperConfig()), li, li)
        analyse(H, name, near_mask=(dmin < 2.0) if dmin.size else None)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
