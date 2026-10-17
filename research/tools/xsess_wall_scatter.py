#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量「同一面墙在多个会话里被放在哪」的离散度。

为什么这是决定性的
------------------
「跨会话位置平均」能不能让 20–31 cm 的墙位置误差按 √N 收敛，前提是：
**各会话对同一面墙的误差彼此独立**。本脚本直接量这个离散度。

难点与解法
----------
会话之间的**全局 gauge 对齐**本身就有误差（033237 留出 1.22 m），直接比位置
会把 gauge 误差混进来。⇒ **在局部窗口内重新对齐**：取一个 3×3 m 的窗口，对两场
的墙点密度图做 2D 互相关找最优平移。gauge 误差在这么小的窗口里近似为常数平移，
会被配准吸收 ⇒ 剩下的就是**局部的墙位置差**，正是要测的量。

（也因此，留出残差大的会话一样能用：033237 留出 1.22 m 不碍事。）

做法
----
1. 每个会话：kf/*.npz 的稠密点云（相机系）→ final_poses 变换到地图系 →
   筛墙高度带 → 累积成 5 cm 的 2D 密度栅格。
2. 两场重叠区域里滑动 3×3 m 窗口，窗口内两场点数都够才用。
3. 窗口内做 2D 互相关（FFT），峰值附近抛物线插值取亚像素平移。
4. 统计所有窗口 × 所有会话对的平移量。

用法
----
    python research/tools/xsess_wall_scatter.py
    python research/tools/xsess_wall_scatter.py --rec A B --win-m 4 --cell 0.05

输出：每对会话的平移量中位 / p25 / p75，以及全体汇总。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REC = ROOT / "navmesh_recordings"

CELL = 0.05          # 栅格边长（m）
WIN_M = 3.0          # 窗口边长（m）
MIN_PTS = 300        # 窗口内单场最少点数（否则没有结构可比）
WALL_LO = 0.30       # 墙高度带下沿（相对地面，m）
WALL_HI = 2.50       # 上沿
MAX_SHIFT = 0.60     # 超过这个平移不认（多半是配到别的墙上了）
NEAR_M = 3.0         # 只用这么近的点（δz = z²/25.5，6 m 处单点噪声就 1.4 m）


def gauge_transform(name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """取 ``xsession/align_<name>_into_*.json`` 里的 gauge 变换（→ 主 gauge）。"""
    d = ROOT / "navmesh_memory" / "wrld_home-7cf435ea" / "xsession"
    for p in sorted(d.glob(f"align_{name}_into_*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        R = j.get("gauge_R_G")
        t = (j.get("gauge") or {}).get("t_G")
        if R is not None and t is not None:
            return np.array(R, float), np.array(t, float)
    return None, None


def load_session(name: str, stride: int, cam_h: float | None = None,
                 half: int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (密度栅格, 栅格原点 xy, 覆盖掩码)。

    ``cam_h`` 留空则从数据估计地面（变换后 z 的低分位数）—— 反正本脚本只量
    **水平**位置，地面只要估得差不多、能把墙的高度带框住就行。
    """
    rec = REC / name
    fp = np.load(rec / "final_poses.npz")
    ids, T = fp["ids"], fp["T"]
    tmap = {int(i): T[k] for k, i in enumerate(ids)}

    # 各会话的地图系原点各自为政 ⇒ **必须**先统一到主 gauge，否则"公共区域"
    # 是错的。（第一次跑就漏了这一步。）
    RG, tG = gauge_transform(name)

    frames = []       # 每帧：世界 xy + 世界 z + 到相机的水平距离
    for _fi, f in enumerate(sorted((rec / "kf").glob("*.npz"))[::max(1, stride)]):
        if half >= 0 and (_fi % 2) != half:
            continue        # 分半对照用：只取奇数帧或偶数帧
        kid = int(f.stem)
        if kid not in tmap:
            continue
        with np.load(f) as z:
            if "pts" not in z:
                continue
            p = z["pts"].astype(np.float64)
        M = tmap[kid]
        R, t = M[:3, :3], M[:3, 3]
        pw = (R @ p.T).T + t
        d = np.linalg.norm(pw[:, :2] - t[:2], axis=1)
        # 只取较近的点：远处深度误差主导（δz = z²/25.5），会淹没墙的位置信号
        m = d < NEAR_M
        if m.sum() == 0:
            continue
        q = pw[m]
        if RG is not None:
            q = (RG @ q.T).T + tG
        frames.append((q[:, :2], q[:, 2]))

    if not frames:
        raise RuntimeError(f"{name}: 没有可用点")

    if cam_h is None:
        zs = np.concatenate([h for _, h in frames])
        ground = float(np.percentile(zs, 2))      # 地面（相机系下方 cam_h）
    else:
        ground = -cam_h

    pts_all = []
    for xy, h in frames:
        rel = h - ground                          # 相对地面的高度
        m = (rel > WALL_LO) & (rel < WALL_HI)
        if m.sum():
            pts_all.append(xy[m])
    if not pts_all:
        raise RuntimeError(f"{name}: 高度带筛完没有点")
    P = np.concatenate(pts_all)
    lo = np.floor(P.min(0) / CELL).astype(int) - 2
    hi = np.ceil(P.max(0) / CELL).astype(int) + 2
    shape = tuple(int(x) for x in (hi - lo))
    g = np.zeros(shape, np.float32)
    ix = np.floor(P[:, 0] / CELL).astype(int) - lo[0]
    iy = np.floor(P[:, 1] / CELL).astype(int) - lo[1]
    np.add.at(g, (ix, iy), 1.0)
    return g, lo.astype(float) * CELL, (g > 0)


def xcorr_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float] | None:
    """a、b 同尺寸窗口，返回 (dx, dy, 峰值归一化相关)；失败返回 None。"""
    a = a - a.mean()
    b = b - b.mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-6 or sb < 1e-6:
        return None
    F = np.fft.rfft2(a) * np.conj(np.fft.rfft2(b))
    c = np.fft.irfft2(F, s=a.shape)
    c = np.fft.fftshift(c)
    n = a.shape[0]
    peak = int(np.argmax(c))
    py, px = divmod(peak, n)
    cy, cx = n // 2, n // 2
    # 抛物线插值取亚像素
    def sub(c0, cm, cp):
        den = (c0 - 2 * cm + cp)
        return 0.0 if abs(den) < 1e-12 else 0.5 * (c0 - cp) / den
    sy = sub(c[(py - 1) % n, px], c[py, px], c[(py + 1) % n, px]) if 0 < py < n - 1 else 0.0
    sx = sub(c[py, (px - 1) % n], c[py, px], c[py, (px + 1) % n]) if 0 < px < n - 1 else 0.0
    dy = (py + sy - cy)
    dx = (px + sx - cx)
    score = float(c[py, px] / (sa * sb * a.size))
    return float(dx), float(dy), score


def compare_pair(ga: np.ndarray, oa: np.ndarray, gb: np.ndarray, ob: np.ndarray,
                 W: int, min_pts: int, max_shift: float) -> np.ndarray:
    """两场栅格在重叠区滑动窗口做互相关，返回 (n, 2) 的平移（米）。"""
    wlo = np.maximum(oa, ob)
    whi = np.minimum(oa + np.array(ga.shape) * CELL,
                     ob + np.array(gb.shape) * CELL)
    sh = (whi - wlo) / CELL
    if np.any(sh < W * 2):
        return np.zeros((0, 2))
    shifts = []
    step = max(1, W // 2)
    ny, nx = int(sh[1]) - W, int(sh[0]) - W
    for yy in range(0, max(ny, 1), step):
        for xx in range(0, max(nx, 1), step):
            i0 = int((wlo[1] - oa[1]) / CELL) + yy
            j0 = int((wlo[0] - oa[0]) / CELL) + xx
            i1 = int((wlo[1] - ob[1]) / CELL) + yy
            j1 = int((wlo[0] - ob[0]) / CELL) + xx
            pa = ga[i0:i0 + W, j0:j0 + W]
            pb = gb[i1:i1 + W, j1:j1 + W]
            if pa.shape != (W, W) or pb.shape != (W, W):
                continue
            if pa.sum() < min_pts or pb.sum() < min_pts:
                continue
            r = xcorr_shift(pa.astype(np.float64), pb.astype(np.float64))
            if r is None:
                continue
            dx, dy, _sc = r
            m = np.hypot(dx, dy) * CELL
            if m > max_shift or m < 1e-6:
                continue
            shifts.append((dx * CELL, dy * CELL))
    return np.array(shifts).reshape(-1, 2)


def main() -> int:
    global CELL, NEAR_M
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rec", nargs="+",
                    default=["20261001_044153", "20261006_001523",
                             "20261010_033237", "20261013_223033"],
                    help="参与比较的录制（需有 final_poses.npz 与 kf/*.npz 的 pts）")
    ap.add_argument("--cam-h", type=float, nargs="+", default=None,
                    help="各录制的相机高（默认全部 1.756；**必须逐场标定**）")
    ap.add_argument("--cell", type=float, default=CELL)
    ap.add_argument("--win-m", type=float, default=WIN_M)
    ap.add_argument("--stride", type=int, default=2, help="关键帧抽样步长")
    ap.add_argument("--min-pts", type=int, default=MIN_PTS)
    ap.add_argument("--max-shift", type=float, default=MAX_SHIFT)
    ap.add_argument("--no-baseline", dest="baseline", action="store_false",
                    help="跳过分半基线（**不建议**：没有它无法解释跨会话那个数）")
    ap.add_argument("--near-m", type=float, default=NEAR_M,
                    help="只用到相机这么近的点（远场深度噪声主导）")
    a = ap.parse_args()
    CELL = a.cell
    NEAR_M = a.near_m
    cams = a.cam_h or [1.756] * len(a.rec)

    grids, origins = {}, {}
    for name, ch in zip(a.rec, cams):
        g, o, _ = load_session(name, a.stride, ch)
        grids[name], origins[name] = g, o
        print(f"  {name}: 栅格 {g.shape}，有效格 {int((g > 0).sum())}，相机高 {ch}")

    W = int(round(a.win_m / CELL))
    print(f"\n窗口 {a.win_m} m（{W}×{W} 格，{CELL * 100:.0f} cm/格），"
          f"单场最少 {a.min_pts} 点\n")
    print(f"{'会话对':<40}{'窗口':>6}{'平移中位 cm':>13}{'p25':>8}{'p75':>8}{'系统偏移 cm':>13}")
    print("-" * 88)

    # ---- 基线：同一会话分半比较 ----
    # 两半共用**同一套位姿** ⇒ 真实墙位置差 = 0，测出来的全是"视角差异 + 方法噪声"。
    # 没有它就没法解释跨会话那个数里有多少是真信号。
    base_mag: list[float] = []
    if a.baseline:
        print("\n基线（同一会话按帧分半，共用一套位姿 ⇒ 真实差应为 0）：")
        print(f"{'会话':<24}{'窗口':>6}{'中位 cm':>10}{'p75':>8}")
        for name in a.rec:
            g0, o0, _ = load_session(name, a.stride, None, half=0)
            g1, o1, _ = load_session(name, a.stride, None, half=1)
            sh = compare_pair(g0, o0, g1, o1, W, a.min_pts, a.max_shift)
            if len(sh) < 5:
                print(f"{name:<24}{len(sh):>6}   （窗口太少）")
                continue
            m = np.linalg.norm(sh, axis=1) * 100
            base_mag.append(float(np.median(m)))
            print(f"{name:<24}{len(sh):>6}{np.median(m):>10.1f}{np.percentile(m, 75):>8.1f}")
        if base_mag:
            print(f"  ⇒ 基线中位 {np.median(base_mag):.1f} cm"
                  f"（视角差异 + 方法噪声，不是真实位置差）")

    all_shifts = []
    for i in range(len(a.rec)):
        for j in range(i + 1, len(a.rec)):
            na, nb = a.rec[i], a.rec[j]
            S = compare_pair(grids[na], origins[na], grids[nb], origins[nb],
                             W, a.min_pts, a.max_shift)
            if len(S) < 5:
                print(f"{na + ' / ' + nb:<40}{len(S):>6}   （窗口太少）")
                continue
            mag = np.linalg.norm(S, axis=1) * 100
            mx, my = np.median(S[:, 0]) * 100, np.median(S[:, 1]) * 100
            all_shifts.append(S)
            print(f"{na + ' / ' + nb:<40}{len(S):>6}{np.median(mag):>13.1f}"
                  f"{np.percentile(mag, 25):>8.1f}{np.percentile(mag, 75):>8.1f}"
                  f"{np.hypot(mx, my):>13.1f}")

    if not all_shifts:
        print("\n没有可用的会话对")
        return 1
    A = np.concatenate(all_shifts)
    mag = np.linalg.norm(A, axis=1) * 100
    print("\n" + "=" * 88)
    print(f"全体：{len(A)} 个窗口，跨会话墙位置离散度")
    print(f"  中位 {np.median(mag):.1f} cm   p25 {np.percentile(mag, 25):.1f}   "
          f"p75 {np.percentile(mag, 75):.1f}   p90 {np.percentile(mag, 90):.1f}")
    print(f"  分量：dx 中位 {np.median(A[:, 0]) * 100:+.1f} ± "
          f"{np.std(A[:, 0]) * 100:.1f}   dy 中位 {np.median(A[:, 1]) * 100:+.1f} ± "
          f"{np.std(A[:, 1]) * 100:.1f} cm")
    print()
    print("判读：")
    print("  · 各向同性、分量中位 ≈ 0 ⇒ 无系统偏移，误差独立 ⇒ √N 成立")
    print("  · 但要扣掉基线：真实位置差 ≈ √(跨会话² − 基线²)（两者独立叠加）")
    if base_mag:
        b = float(np.median(base_mag)); t = float(np.median(mag))
        sig = max(t * t - b * b, 0.0) ** 0.5
        print(f"\n  跨会话 {t:.1f} cm  基线（视角+方法噪声）{b:.1f} cm"
              f"  ⇒ 真实墙位置差 ≈ {sig:.1f} cm")
        print(f"  ⇒ {len(a.rec)} 场平均后 → {sig / np.sqrt(len(a.rec)):.1f} cm"
              f"（若误差独立）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
