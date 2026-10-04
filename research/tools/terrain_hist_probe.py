# -*- coding: utf-8 -*-
r"""实验：逐格**高度直方图**能不能分开"楼梯"与"平地"？

    python research/tools/terrain_hist_probe.py 20261001_044153

问题
----
``nav_mapping`` 每格只留 7 个数：4 个分带计数 + ``n/Σh/Σh²`` 三个矩。
矩只能答"有没有一张平整的面、多高"，答不了"这张面是几级台阶"——
周期性是**格内双峰**，标量装不下。``surface_counts`` 的 docstring 已经点名：
*"判『面正好在 3.0 m、能不能站上去』不够——那要众数，也就是带内高度直方图，尚未做。"*

本脚本回答的是：**补上直方图之后，楼梯信号到底在不在？**

高度定义
--------
**不自己发明。** 直接取 ``mapper._rot[k]`` 里的 ``rel``/``cnt``，套 mapper 本人的公式
（``nav_mapping.py:562-564``）::

    h = rel + cam_h
    h = h - _ground_correction(h, cnt, rxy, ...)

这样量到的 h 与线上分带判定**逐位同源**。用自己那套会得到"测的不是同一件事"的结论。

判据
----
楼梯在格内的签名是**双峰**（踏面一个、踢面一个），峰间距 = 踢面高 0.05–0.25 m；
平地是单峰。所以特征取：

* ``peak2``  次峰高度 − 主峰高度（无次峰记 NaN）
* ``n_modes`` 平滑后显著峰的个数
* ``mode_h`` 主峰高度（对照现有偏高的均值 ``mean_h``）

不靠人工标注先验：先看**信号在不在**。若全图找不到"双峰且峰间距落在踢面尺度"的格，
说明该录制里没有可分离的楼梯，补直方图也换不来东西——这本身就是结论。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "research" / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "research" / "tools"))

from backend.nav_mapping import _ground_correction          # noqa: E402
from backend.nav_online import OnlineNavConfig              # noqa: E402
from mapping_gate import recorded_config, replay            # noqa: E402

BIN = 0.025          # 2.5 cm：能分辨 5 cm 的最小踢面，又不至于被噪声灌满
H_LO, H_HI = -0.60, 1.20
NBIN = int(round((H_HI - H_LO) / BIN))
RISER_LO, RISER_HI = 0.05, 0.30      # 虚报前的楼梯踢面尺度（VRChat 常见）


def cell_hists(rec: Path, cfg) -> tuple[np.ndarray, tuple[int, int], object, np.ndarray]:
    """把所有关键帧的 h 累进 NavGrid 布局的逐格直方图。

    返回 ``(hist, (x0,y0), mapper, min_range)``。``min_range`` 是每格**最近一次
    被观测到时的距离**（世界米）——``rxy`` 本身就是相机系坐标，``|rxy|`` 即量程，
    不需要额外算。这是"这个格的高度是从多远的地方看出来的"，是判断高度精度
    能否支撑台阶判别的关键自变量。取 min 而不是 mean：远处的一次误观测不该
    把一个近处被反复确认的格拖进远档。
    """
    ng, _trail, _st, mapper = replay(rec, cfg)
    c = cfg
    ix0, iy0 = mapper._grid_lo
    gh, gw = ng.grid.shape
    hmap = np.zeros((gh, gw, NBIN), np.float32)
    wmap = np.zeros((gh, gw), np.float32)
    rng = np.full((gh, gw), np.inf, np.float32)
    for k, (rxy, rel, cnt, _R) in mapper._rot.items():
        h = rel + np.float32(mapper.cam_h)
        h = h - np.float32(_ground_correction(h, cnt, rxy, c.ground_offset_max_m,
                                              c.ground_offset_min_range_m,
                                              c.ground_plane_max_deg, c.ground_plane_band_m))
        # 关键帧点本来就在 base 系原点附近的小方块内；用 mapper 自己的 res_m/原点。
        xy = rxy.astype(np.float64) + mapper._pose[k][:2, 3]
        ix = np.floor(xy[:, 0] / c.res_m).astype(np.int64)
        iy = np.floor(xy[:, 1] / c.res_m).astype(np.int64)
        gx, gy = ix - ix0, iy - iy0
        keep = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh) & (h >= H_LO) & (h < H_HI)
        if not keep.any():
            continue
        flat = gy[keep] * gw + gx[keep]
        b = ((h[keep] - H_LO) / BIN).astype(np.int64)
        w = np.asarray(cnt, np.float32)[keep]
        # (cell, bin) 联合 bincount —— 比两层循环快一个量级
        hmap += np.bincount(flat * NBIN + b, w, minlength=gh * gw * NBIN).reshape(gh, gw, NBIN)
        wmap += np.bincount(flat, w, minlength=gh * gw).reshape(gh, gw)
        r = np.hypot(rxy[keep, 0], rxy[keep, 1]).astype(np.float32) * c.world_scale
        np.minimum.at(rng.reshape(-1), flat, r)
    return hmap, (ix0, iy0), mapper, rng


def peaks(h: np.ndarray, min_frac: float = 0.25, min_gap_bins: int = 2) -> tuple[float, float, int]:
    """返回 (主峰高度, 次峰高度或 NaN, 显著峰个数)。峰高按直方图总重的比例算。"""
    tot = float(h.sum())
    if tot <= 0:
        return np.nan, np.nan, 0
    sm = cv2.GaussianBlur(h.reshape(1, -1), (5, 1), 0).reshape(-1)
    thr = min_frac * float(sm.max())
    idx = np.flatnonzero(sm >= thr)
    if idx.size == 0:
        return np.nan, np.nan, 0
    # 把相邻的连续段并成一个峰
    groups, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= min_gap_bins:
            cur.append(b)
        else:
            groups.append(cur); cur = [b]
    groups.append(cur)
    ranked = sorted(groups, key=lambda g: float(sm[g].max()), reverse=True)
    def _h(g):
        return float(H_LO + (g[int(np.argmax(sm[g]))] + 0.5) * BIN)
    second = _h(ranked[1]) if len(ranked) > 1 else np.nan
    return _h(ranked[0]), second, len(ranked)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rec", nargs="?", default="20261001_044153")
    ap.add_argument("--min-points", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=ROOT / "tmp")
    args = ap.parse_args()

    rec = ROOT / "navmesh_recordings" / args.rec
    cfg, cinfo = recorded_config(rec)
    hmap, _lo, _m, _rng = cell_hists(rec, cfg)
    hmap, wmap = hmap, hmap.sum(axis=2)
    gh, gw = hmap.shape[:2]

    big = wmap >= args.min_points
    sep = np.full((gh, gw), np.nan, np.float32)
    nmode = np.zeros((gh, gw), np.int16)
    mh = np.full((gh, gw), np.nan, np.float32)
    for r in range(gh):
        for c in range(gw):
            if not big[r, c]:
                continue
            a, b, n = peaks(hmap[r, c])
            mh[r, c], nmode[r, c] = a, n
            if n >= 2 and np.isfinite(b):
                sep[r, c] = abs(b - a)

    stair = big & (nmode >= 2) & np.isfinite(sep) & (sep >= RISER_LO) & (sep <= RISER_HI)
    print(f"\n=== {args.rec}  range_m={cinfo['range_m']} ===")
    print(f"格子总数 {gh}x{gw}={gh*gw}，有效格(点数≥{args.min_points:g}) {int(big.sum())}")
    print(f"主峰高度可读: {int(np.isfinite(mh).sum())}  格")
    print(f"多峰格(≥2峰): {int((big & (nmode>=2)).sum())}")
    print(f"★ 疑似楼梯格(≥2峰 且 峰间距 {RISER_LO}-{RISER_HI} m): {int(stair.sum())}")
    if stair.any():
        s = sep[stair]
        print(f"    峰间距 分位 p10/p50/p90 = {np.nanpercentile(s,10):.3f} / "
              f"{np.nanpercentile(s,50):.3f} / {np.nanpercentile(s,90):.3f} m")
        print(f"    主峰高度 中位 = {np.nanmedian(mh[stair]):+.3f} m")
        # 对照：非楼梯格里，峰间距落在踢面尺度的有多少
        rest = big & ~stair & (nmode >= 2) & np.isfinite(sep)
        if rest.any():
            r = sep[rest]
            print(f"    对照(其余多峰格 {int(rest.sum())}) 峰间距中位 = {np.nanmedian(r):.3f} m，"
                  f"落在踢面尺度的 {int(((r>=RISER_LO)&(r<=RISER_HI)).sum())} 格 "
                  f"({100.0*((r>=RISER_LO)&(r<=RISER_HI)).mean():.1f}%)")
    else:
        print("    —— 全图找不到。**要么该录制没有楼梯，要么格内直方图救不回来。**")

    args.out.mkdir(parents=True, exist_ok=True)
    vis = np.zeros((gh, gw, 3), np.uint8)
    vis[..., 0] = np.where(big, 40, 0) + np.where(stair, 0, 0)
    vis[..., 1] = np.where(big, 40, 0)
    vis[..., 2] = np.where(big, 40, 0) + np.where(stair, 215, 0)
    sc = min(3.0, 1400 / max(gh, gw))
    if sc > 1.0:
        vis = cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    p = args.out / f"terrain_hist_{args.rec}.png"
    cv2.imwrite(str(p), vis)
    print(f"\n图已写出：{p}  （白=有效格，红=疑似楼梯）")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
