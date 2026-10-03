# -*- coding: utf-8 -*-
"""只用矩（A 方案）能不能把"面"和"杂物"分开？

    python -m tools.band_moments [录制名]

A 方案每个 (格,带) 只存 n / Σh / Σh² 三个可加量 ⇒ 均值高度 + 方差。
本脚本按这个最小实现跑一遍，看它够不够：

  1. 每格均值高度  →  3×3 邻域最小二乘拟合 h = a + b·x + c·y
  2. 逐格残差 = mean − 拟合值        ⇒ 平面格残差小
  3. 方差 = E[h²] − mean²            ⇒ 竖直杂物/斜面方差大
  4. 残差分布是不是**双峰**（面 vs 杂物）？双峰说明矩就能分开，
     单峰说明必须上直方图（B）才能拆开同格的两个面。

另外直接对比 A 与 B 在同一批数据上的一致性：众数高度 vs 均值高度
在"这个格有没有一个干净的面"上给出的判断是否相同。
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
CELL_M = 0.3
MIN_PTS = 20
PLANE_TOL = 0.15          # 残差小于它算"这一格贴平面"


def moments(frames, m: KeyframeGridMapper):
    """头顶 ``obst_top_m`` 以上的点，按粗格累 (n, Σh, Σh², 最小观测距离)。

    这就是 A 方案的全部信息量——三个可加量（第四个只是给分析用的诊断量，不进累加器）。
    """
    c = m.cfg
    base = (0, 0)
    ny = nx = 0
    acc = None

    def slot(size: int) -> list[np.ndarray]:
        nonlocal acc
        if acc is None or acc[0].size != size:
            acc = [np.zeros(size) for _ in range(4)]
        return acc

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
        if acc is None:
            base = (int(gy.min()), int(gx.min()))
            ny = int(gy.max()) - base[0] + 1
            nx = int(gx.max()) - base[1] + 1
        ok = (h > c.obst_top_m) & (gy >= base[0]) & (gy < base[0] + ny) & \
             (gx >= base[1]) & (gx < base[1] + nx)
        if not ok.any():
            continue
        cell = (gy[ok] - base[0]) * nx + (gx[ok] - base[1])
        w = cn[ok].astype(np.float64)
        hh = h[ok].astype(np.float64)
        d = np.hypot(rxy[ok, 0], rxy[ok, 1]).astype(np.float64)
        g = slot(ny * nx)
        g[0] += np.bincount(cell, weights=w, minlength=ny * nx)
        g[1] += np.bincount(cell, weights=w * hh, minlength=ny * nx)
        g[2] += np.bincount(cell, weights=w * hh * hh, minlength=ny * nx)
        # 最小观测距离逐点取 min（诊断量，不进累加器）
        np.minimum.at(g[3], cell, d)
    if acc is None:
        z = np.zeros((0, 0))
        return z, z, z, z
    g = acc
    n = g[0].reshape(ny, nx)
    sh = g[1].reshape(ny, nx)
    sh2 = g[2].reshape(ny, nx)
    dm = g[3].reshape(ny, nx)
    return n, sh, sh2, dm


def local_plane_residual(p: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """3×3 邻域最小二乘拟合 h = a + b·gx + c·gy，返回逐格残差。"""
    K = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                  [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [1.0, -1.0, 1.0],
                  [1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0]])
    Hh, Ww = p.shape
    res = np.full((Hh, Ww), np.nan)
    for r in range(1, Hh - 1):
        for cc in range(1, Ww - 1):
            win = p[r - 1:r + 2, cc - 1:cc + 2].reshape(9)
            m = valid[r - 1:r + 2, cc - 1:cc + 2].reshape(9)
            if m.sum() < 6:
                continue
            A = K[m]
            y = win[m]
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            res[r, cc] = p[r, cc] - sol[0]
    return res


def main() -> None:
    rec_name = sys.argv[1] if len(sys.argv) > 1 else REC
    frames = load(ROOT / "navmesh_recordings" / rec_name)
    if not frames:
        print("无关键帧")
        return
    m = KeyframeGridMapper(MapperConfig())
    n, sh, sh2, dmin = moments(frames, m)
    valid = n >= MIN_PTS
    mean = np.where(valid, sh / np.maximum(n, 1), np.nan)
    var = np.where(valid, np.maximum(sh2 / np.maximum(n, 1) - mean ** 2, 0.0), np.nan)
    print(f"录制 {rec_name}  {len(frames)} 关键帧  粗格 {CELL_M} m")
    print(f"头顶 2 m 以上的格：参与统计 {int(valid.sum())}  均值高度 "
          f"{np.nanpercentile(mean, [5, 50, 95]).round(2).tolist()} m")
    print(f"  方差（标准差 m）中位 {np.nanmedian(np.sqrt(var)):.3f}  p90 {np.nanpercentile(np.sqrt(var), 90):.3f}")
    res = local_plane_residual(mean, valid)
    rv = res[np.isfinite(res)]
    print(f"  3×3 局部平面残差：中位 {np.median(rv):.3f} m  p10 {np.percentile(rv, 10):.3f}  "
          f"p90 {np.percentile(rv, 90):.3f}")
    print(f"  残差 < {PLANE_TOL} m 的格占比 {100.0 * float((np.abs(rv) < PLANE_TOL).mean()):.1f}%")
    # 残差分布是否双峰（面 vs 杂物）？用 0.1 m 桶画直方图
    hist, edges = np.histogram(rv, bins=np.arange(-0.6, 0.61, 0.1))
    peak = int(np.argmax(hist))
    print("\n  残差直方图（0.1 m 桶，%）：")
    for a, cnt in zip(edges[:-1], hist):
        pct = 100.0 * cnt / max(hist.sum(), 1)
        if pct < 0.5:
            continue
        mark = " ← 峰" if a == edges[peak] else ""
        print(f"     {a:+.1f} .. {a + 0.1:+.1f} m {pct:5.1f}% {'█' * int(pct / 2)}{mark}")
    # A 的两个量联合起来能不能分
    planar = (np.abs(res) < PLANE_TOL)
    thin = (var < np.nanmedian(var) * 1.0)
    both = np.isfinite(res) & valid
    print(f"\n  「残差小」且「方差小」的格 {int((planar & thin & both).sum())} / "
          f"{int(both.sum())}（{100.0 * (planar & thin & both).sum() / max(int(both.sum()), 1):.1f}%）"
          f" —— 这些是干净的水平面候选")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
