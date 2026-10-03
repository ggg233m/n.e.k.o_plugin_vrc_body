# -*- coding: utf-8 -*-
"""导航尺子 + 常数审计。**不改变任何行为**，纯观测。

    python -m tools.nav_audit [录制名...]

## 为什么要两个尺子

现有唯一的尺子是"走过∩OCC / 走过格数"。它只惩罚**路径上**的误判，
对"路径外的真结构被清掉"权重为零——沙发后面那道 0.85 m 矮墙（496 格）就是这么藏住的：
用户不会去穿一道矮墙，所以这把尺量不到它。

    尺子 1（现有）  走过∩OCC 率          抓：路径上的虚假障碍
    尺子 2（新增）  漏放格中的近距平面   抓：路径外的真结构被清掉

**单把必然被骗**：尺子 1 单独用会把人往"放宽阈值、降低路径误判"的方向推，
正好把尺子 2 变坏。任何改动必须两个数同时看。

## 尺子 2 的判据（每一条都单独验过）

    近距      观测距离中位 < q_near_m。δz = z²/25.5，1.5 m 处 16 cm，
              地面散点抖不出 0.85 m，所以近距高面不可能是噪声。
    平面      3×3 邻域众数高度极差 ≤ 0.15 m。随机置换对照 0 格通过（16×/30× 信噪比）。
    多层      ≥2 个 z 层有打中。单层命中可能只是量化抖动。
    量够      带内加权点数 ≥ MIN_PTS。

四条同时成立 ⇒ 该格大概率有一张**真实水平面**，不管它在栅格里被判成什么。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import (KeyframeGridMapper, MapperConfig,      # noqa: E402
                                 _ground_correction)
from backend.nav_grid import FREE, OCC, UNK                               # noqa: E402
from tools.precision_probe import load                                   # noqa: E402

HBIN = 0.05
MIN_PTS = 20
PLANAR_TOL = 0.15
MIN_NB = 6
MIN_HIT_LAYERS = 2
RECS = ("20261001_044153", "20260929_045615", "20261001_044120")


def band_pass(m: KeyframeGridMapper, c: MapperConfig, n_frames: int):
    """一遍扫帧，攒 0.3~2.0 m 带的 (众数高度, 加权点数, 加权距离和)。"""
    H, W = m._acc_g.shape
    lo = 12.0
    cache = []
    for i in range(n_frames):
        r = m._rot.get(i)
        if r is None or r[0] is None or not len(r[0]):
            continue
        rxy, rel, cnt, _R = r
        h = rel + np.float32(m.cam_h)
        h = h - np.float32(_ground_correction(h, cnt, rxy, c.ground_offset_max_m,
                                              c.ground_offset_min_range_m,
                                              c.ground_plane_max_deg, c.ground_plane_band_m))
        ok = (h > c.ground_tol_m) & (h <= c.obst_top_m)
        if not ok.any():
            continue
        xy = rxy[ok].astype(np.float64) + m._pose[i][:2, 3]
        ax, ay = m._acc_lo
        ix = np.floor(xy[:, 0] / c.res_m).astype(np.int64) - ax
        iy = np.floor(xy[:, 1] / c.res_m).astype(np.int64) - ay
        good = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if not good.any():
            continue
        cam = m._pose[i][:2, 3]
        rng = np.hypot(xy[:, 0] - cam[0], xy[:, 1] - cam[1])
        cache.append((ix[good], iy[good], h[ok][good], cnt[ok][good], rng[good]))
        lo = min(lo, float(cache[-1][2].min()))
    if not cache:
        return None
    nh = int((c.obst_top_m - lo) / HBIN) + 2
    hist = np.zeros((nh, H, W), np.float32)
    rsum = np.zeros((H, W), np.float32)
    for ix, iy, h, w, rng in cache:
        hb = np.floor((h - lo) / HBIN).astype(np.int64)
        good = (hb >= 0) & (hb < nh)
        if not good.any():
            continue
        flat = iy[good] * W + ix[good]
        hist += np.bincount(hb[good] * (H * W) + flat, weights=w[good],
                            minlength=nh * H * W).reshape(nh, H, W).astype(np.float32)
        rsum += np.bincount(flat, weights=w[good] * rng[good], minlength=H * W).reshape(H, W).astype(np.float32)
    n = hist.sum(axis=0)
    mode_h = np.where(n > 0, lo + hist.argmax(axis=0) * HBIN, np.nan)
    mean_r = np.where(n > 0, rsum / np.maximum(n, 1e-9), np.nan)
    return mode_h, n, mean_r


def _spread(mode_h, valid):
    M = np.where(valid, mode_h, np.nan)
    H, W = M.shape
    stack = np.full((9, H, W), np.nan)
    for k, (dy, dx) in enumerate([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]):
        ys0, ys1 = max(0, dy), H + min(0, dy)
        xs0, xs1 = max(0, dx), W + min(0, dx)
        stack[k, ys0:ys1, xs0:xs1] = M[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    with np.errstate(invalid="ignore"):
        return np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0), np.sum(~np.isnan(stack), axis=0)


class Spy(KeyframeGridMapper):
    def _by_quality(self, base_occ, n_g, n_o, o_n, o_m, g_n, o_mk):
        occ, restored, rejected = super()._by_quality(base_occ, n_g, n_o, o_n, o_m, g_n, o_mk)
        self.spy = {"base_occ": base_occ.copy(), "n_g": n_g.copy(), "n_o": n_o.copy(),
                    "o_n": o_n.copy(), "o_m": o_m.copy(), "o_mk": o_mk.copy(),
                    "occ": occ.copy(), "restored": restored.copy(), "rejected": rejected.copy()}
        return occ, restored, rejected


def walked_line(ng, walked) -> np.ndarray:
    line = np.zeros(ng.grid.shape, np.uint8)
    for poly in walked:
        pts = np.array([ng.to_cell(p)[::-1] for p in np.asarray(poly, float)], np.int32)
        if len(pts) >= 2:
            cv2.polylines(line, [pts.reshape(-1, 1, 2)], False, 1, 1)
    return line


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def audit_consts(c: MapperConfig, s: dict, hp: tuple, m: KeyframeGridMapper):
    """量每个阈值常数实际面对的分布，报常数落在哪个百分位。"""
    n_g, n_o, o_n, o_m = s["n_g"], s["n_o"], s["o_n"], s["o_m"]
    nlay_hit = None
    if m._ray_hit is not None:
        hz = np.stack([m._cut_acc(m._ray_hit[z]) for z in range(len(m._ray_hit))])
        mz = np.stack([m._cut_acc(m._ray_mis[z]) for z in range(len(m._ray_mis))])
        nlay_hit = (hz > 0).sum(axis=0)
    rows = []

    # --- min_pts：作用在 n_o > 0 的格上 ---
    v = n_o[n_o > 0]
    rows.append(("min_pts", c.min_pts, "n_o>0 的格的 n_o",
                 f"p25 {pct(v, 25):.0f}  中位 {np.median(v):.0f}  p75 {pct(v, 75):.0f}",
                 f"落在 p{100 * (v < c.min_pts).mean():.0f}——"
                 f"{(v < c.min_pts).mean() * 100:.0f}% 的有票格直接被这一个数筛掉"))

    # --- occ_ground_ratio：作用在 n_o > min_pts 的格上 ---
    sel = n_o > c.min_pts - 0.5
    r = n_o[sel] / np.maximum(n_g[sel], 1e-9)
    rows.append(("occ_ground_ratio", c.occ_ground_ratio, "n_o>min_pts 的格的 n_o/n_g",
                 f"p25 {pct(r, 25):.2f}  中位 {np.median(r):.2f}  p75 {pct(r, 75):.2f}",
                 f"落在 p{100 * (r < c.occ_ground_ratio).mean():.0f}——"
                 f"{(r < c.occ_ground_ratio).mean() * 100:.0f}% 被地面压制杀掉"))

    # --- ray_beta：作用在"某层有打中"的格的该层上 ---
    if nlay_hit is not None:
        best = np.argmax(hz, axis=0)
        hb = np.take_along_axis(hz, best[None], axis=0)[0]
        mb = np.take_along_axis(mz, best[None], axis=0)[0]
        has = hb > 0
        ratio = mb[has] / hb[has]
        rows.append(("ray_beta", c.ray_beta, "有打中的层：看穿/打中",
                     f"p5 {pct(ratio, 5):.1f}  中位 {np.median(ratio):.1f}  p95 {pct(ratio, 95):.1f}",
                     f"落在 p{100 * (ratio < c.ray_beta).mean():.0f}——"
                     f"{(ratio < c.ray_beta).mean() * 100:.0f}% 的格恒触发看穿清零"))
    else:
        rows.append(("ray_beta", c.ray_beta, "—", "（ray_clear 关着）", "—"))

    # --- q_solo_pts / q_far_tol：作用在被降级的格上 ---
    rej = s["rejected"] & ~s["restored"]
    if rej.any():
        far = np.maximum(n_o - o_n - o_m, 0.0)
        frac = (far / np.maximum(n_o, 1.0))[rej]
        rows.append(("q_solo_pts", c.q_solo_pts, "被降级格的 n_o",
                     f"p25 {pct(n_o[rej], 25):.0f}  中位 {np.median(n_o[rej]):.0f}  "
                     f"p75 {pct(n_o[rej], 75):.0f}",
                     f"落在 p{100 * (n_o[rej] < c.q_solo_pts).mean():.0f}"))
        rows.append(("q_far_tol", c.q_far_tol, "被降级格的远带票占比",
                     f"p25 {pct(frac, 25):.2f}  中位 {np.median(frac):.2f}  "
                     f"p75 {pct(frac, 75):.2f}",
                     f"落在 p{100 * (frac <= c.q_far_tol).mean():.0f}——"
                     f"{(frac <= c.q_far_tol).mean() * 100:.0f}% 能走单帧例外"))
    else:
        rows.append(("q_solo_pts", c.q_solo_pts, "被降级格的 n_o", "没有格被降级", "—"))
        rows.append(("q_far_tol", c.q_far_tol, "被降级格的远带票占比", "没有格被降级", "—"))

    # --- ground_tol_m：作用在带沿 ---
    mode_h, n_pts, mean_r = hp
    near_edge = np.isfinite(mode_h) & (mode_h < c.ground_tol_m + 0.2)
    tot = int(np.isfinite(mode_h).sum())
    rows.append(("ground_tol_m", c.ground_tol_m, "众数高度落在带沿 0.2 m 内的格",
                 f"{int(near_edge.sum())} / {tot} 格",
                 f"{near_edge.sum() / max(tot, 1) * 100:.1f}% 的面紧贴带沿——带沿埋在噪声里"))
    rows.append(("q_near_m", c.q_near_m, "δz = z²/25.5（推导，非拟合）",
                 f"1.5 m→{1.5 ** 2 / 25.5:.2f} m  3 m→{3 ** 2 / 25.5:.2f} m  "
                 f"5 m→{5 ** 2 / 25.5:.2f} m",
                 f"障碍带宽 {c.obst_top_m - c.ground_tol_m:.1f} m ⇒ 1.5 m 外散射是必然的，"
                 f"这条有依据"))
    return rows


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
        ng.build(radius_m=0.25, walked=m.walked())
        g = ng.grid
        print(f"\n=== {name} ===  {len(frames)} 关键帧  栅格 {g.shape}  res {c.res_m} m")
        print(f"  OCC {int((g == OCC).sum())}  FREE {int((g == FREE).sum())}  "
              f"UNK {int((g == UNK).sum())}")

        # ---------- 尺子 1 ----------
        line = walked_line(ng, m.walked())
        occ = g == OCC
        ow = int((occ & (line > 0)).sum())
        nl = int((line > 0).sum())
        print(f"\n  尺子 1  走过∩OCC 率 = {ow}/{nl} = {ow / max(nl, 1) * 100:.2f}%")

        # ---------- 尺子 2 ----------
        hp = band_pass(m, c, len(frames))
        if hp is None:
            print("  尺子 2  障碍带里没有数据")
            continue
        mode_h, n_pts, mean_r = hp
        g_mode = m._cut_acc(np.nan_to_num(mode_h, nan=0.0))
        g_n = m._cut_acc(n_pts)
        g_r = m._cut_acc(np.nan_to_num(mean_r, nan=0.0))
        valid = g_n >= MIN_PTS
        spread, n_ok = _spread(g_mode, valid)
        planar = valid & (n_ok >= MIN_NB) & (spread <= PLANAR_TOL)
        if m._ray_hit is not None:
            hz = np.stack([m._cut_acc(m._ray_hit[z]) for z in range(len(m._ray_hit))])
            nlay = (hz > 0).sum(axis=0)
        else:
            nlay = np.zeros(g.shape, np.int32)
        cand = planar & (g_r < c.q_near_m) & (nlay >= MIN_HIT_LAYERS)
        idx = np.flatnonzero(cand.ravel())
        st = g.ravel()[idx]
        n_free = int((st == FREE).sum())
        n_unk = int((st == UNK).sum())
        n_occ = int((st == OCC).sum())
        print(f"  尺子 2  近距平面证据格 {len(idx)}（平面 ∧ 距离<{c.q_near_m} m ∧ "
              f"≥{MIN_HIT_LAYERS} 个打中层）")
        print(f"          其中 OCC {n_occ}  **FREE {n_free}**  UNK {n_unk}"
              f"   ⇒ 漏放 {n_free + n_unk} 格")
        if len(idx):
            hh = g_mode.ravel()[idx]
            print(f"          高度中位 {np.median(hh):.2f} m   距离中位 "
                  f"{np.median(g_r.ravel()[idx]):.2f} m")

        # ---------- 常数审计 ----------
        s = {k: (v[::-1] if v.ndim == 2 else v) for k, v in m.spy.items()}
        rows = audit_consts(c, s, (mode_h, n_pts, mean_r), m)
        print(f"\n  常数审计（{c.min_pts=}  {c.occ_ground_ratio=}  {c.ray_beta=}）")
        for k, v, dist, stat, verdict in rows:
            print(f"    {k:20s} = {str(v):>6s}  {dist}")
            print(f"    {'':20s}   {stat}")
            print(f"    {'':20s}   → {verdict}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
