# -*- coding: utf-8 -*-
"""这些面到底是"哪几层"？——共现范围 + 倾角，用来分楼板与天花板。

    python -m tools.band_levels [录制名]

## 要回答的

`band_height_hist` 量出三张连贯面：2.55 m（连贯 69%）、3.88 m（92%）、4.62 m（92%）。
要判"楼板还是天花板"，得知道它们的**空间关系**：

  * 同一批格子上同时出现两张面 ⇒ 大概率是同一个房间的**楼板 + 天花板**（成对）
  * 一张的范围明显小 ⇒ 那是**物件**（栏杆/灯架/人），不是一层
  * 倾角差很多 ⇒ 一张是斜面（坡道/楼梯），一张是平的

## 量什么

每条带每格：众数高度 ``mode_h``、点数 ``n``、以及该带内高度的 5/95 分位（厚度）。

    覆盖      每带有多少格
    共现      任意两带的格集交集占比（0.9 = 几乎完全重合 ⇒ 成对）
    倾角      对 mode_h 做最小二乘平面拟合 h = a + b·gx + c·gy，报 b/c 换算成每米的升降
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
HBIN = 0.05
CELL_M = 0.3
MIN_PTS = 20
BANDS = ("below", "lo", "mid", "hi")


def band_surface(frames, lo_i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一条带每格的 (mode_h, 厚度 p95−p5, 有效掩码)。"""
    H, _ny, _nx, h0 = _abs_hist(frames, lo_i)          # h0 = 直方图的高度下界，**必须加回来**
    n = H.sum(axis=0)
    valid = n >= MIN_PTS
    mode = H.argmax(axis=0)
    cs = np.cumsum(H, axis=0)
    tot = np.maximum(cs[-1], 1.0)
    p05 = (cs >= 0.05 * tot).argmax(axis=0)
    p95 = (cs >= 0.95 * tot).argmax(axis=0)
    scale = np.full(H.shape[1:], np.nan)
    scale[valid] = h0 + mode[valid] * HBIN
    thick = np.full(H.shape[1:], np.nan)
    thick[valid] = (p95[valid] - p05[valid]) * HBIN
    return scale, thick, valid


def _band_mask(h: np.ndarray, c, lo_i: int) -> np.ndarray:
    if lo_i == 0:
        return h <= -c.ground_tol_m
    if lo_i == 1:
        return (h > c.obst_top_m) & (h <= c.hi_band_m)
    if lo_i == 2:
        return (h > c.hi_band_m) & (h <= c.hi_band_top_m)
    return h > c.hi_band_top_m


def _abs_hist(frames, lo_i: int):
    """绝对高度桶的联合直方图 (nh, ny, nx)。两遍：第一遍定网格与 hbin 下界，第二遍累。
    两遍各用一个 mapper——``add_keyframe`` 会拒绝重复 id，同一个实例不能用两遍。"""
    c = MapperConfig()
    base = (0, 0)
    ny = nx = 0
    lo = 12.0

    def heights(m, i, fr):
        """第 i 帧的 (h, w, cell, gx, gy)——h 已做地面修正。"""
        m.add_keyframe(i, fr[1], fr[2])
        m._rotated(i)
        rxy, rel, cn, _R = m._rot[i]
        if rxy is None or not len(rxy):
            return None
        h = rel + np.float32(m.cam_h)
        h = h - np.float32(_ground_correction(h, cn, rxy, c.ground_offset_max_m,
                                              c.ground_offset_min_range_m,
                                              c.ground_plane_max_deg, c.ground_plane_band_m))
        xy = rxy.astype(np.float64) + m._pose[i][:2, 3]
        gx = np.floor(xy[:, 0] / CELL_M).astype(np.int64)
        gy = np.floor(xy[:, 1] / CELL_M).astype(np.int64)
        return h, cn.astype(np.float64), gx, gy

    # 第一遍：网格范围 + hbin 下界
    m1 = KeyframeGridMapper(c)
    for i, fr in enumerate(frames):
        r = heights(m1, i, fr)
        if r is None:
            continue
        h, _w, gx, gy = r
        if ny == 0:
            base = (int(gy.min()), int(gx.min()))
            ny, nx = int(gy.max()) - base[0] + 1, int(gx.max()) - base[1] + 1
        sel = _band_mask(h, c, lo_i)
        if sel.any():
            lo = min(lo, float(h[sel].min()))
    nh = int((12.0 - lo) / HBIN) + 2
    H = np.zeros((nh, ny, nx))
    m2 = KeyframeGridMapper(c)
    for i, fr in enumerate(frames):
        r = heights(m2, i, fr)
        if r is None:
            continue
        h, w, gx, gy = r
        ok = _band_mask(h, c, lo_i)
        ok &= (gy >= base[0]) & (gy < base[0] + ny) & (gx >= base[1]) & (gx < base[1] + nx)
        if not ok.any():
            continue
        cell = (gy[ok] - base[0]) * nx + (gx[ok] - base[1])
        hb = np.floor((h[ok] - lo) / HBIN).astype(np.int64)
        good = (hb >= 0) & (hb < nh)
        if not good.any():
            continue
        H += np.bincount(hb[good] * (ny * nx) + cell[good], weights=w[ok][good],
                         minlength=nh * ny * nx).reshape(nh, ny, nx)
    return H, ny, nx, lo


def tilt(p: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """对 mode_h 做全局最小二乘平面拟合，返回每米的升降 (b, c)。"""
    ys, xs = np.nonzero(valid)
    if len(ys) < 50:
        return float("nan"), float("nan")
    A = np.column_stack([np.ones(len(ys)), xs * CELL_M, ys * CELL_M])
    sol, *_ = np.linalg.lstsq(A, p[ys, xs], rcond=None)
    return float(sol[1]), float(sol[2])


def main() -> None:
    rec_name = sys.argv[1] if len(sys.argv) > 1 else REC
    frames = load(ROOT / "navmesh_recordings" / rec_name)
    if not frames:
        print("无关键帧")
        return
    print(f"录制 {rec_name}  {len(frames)} 关键帧  统计格 {CELL_M} m")
    S: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for li, name in enumerate(BANDS):
        mode_h, thick, valid = band_surface(frames, li)
        if not valid.any():
            print(f"\n[{name}] 没有足够的点，跳过")
            continue
        S[name] = (mode_h, thick, valid)
        v = mode_h[valid]
        b, c = tilt(mode_h, valid)
        print(f"\n[{name}] 覆盖 {int(valid.sum())} 格   mode_h 中位 {np.median(v):.2f} m "
              f"(p10..p90 {np.percentile(v, 10):.2f}..{np.percentile(v, 90):.2f})")
        print(f"      带内厚度 p50 {np.nanmedian(thick):.2f} m   平面倾角 b={b:+.3f} c={c:+.3f} m/m"
              f"（合 {np.hypot(b, c):.3f}）")
    names = list(S)
    if len(names) < 2:
        return
    print("\n共现（交集 / 较小的那个集合的大小；→1.0 = 同一批格子，大概率是楼板+天花板成对）")
    print(f"{'':8s}" + "".join(f"{n:>9s}" for n in names))
    for a in names:
        row = f"{a:8s}"
        for b in names:
            if a == b:
                row += f"{'-':>9s}"
                continue
            A, B = S[a][2], S[b][2]
            inter = int((A & B).sum())
            row += f"{inter / max(min(int(A.sum()), int(B.sum())), 1):>9.2f}"
        print(row)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()

