# -*- coding: utf-8 -*-
"""2 m 以下的水平面，现在到底算不算障碍？——低天花板的 OCC/FREE 交叉表。

    python -m tools.low_ceiling_nav [录制名...]

## 要回答的

``low_ceiling_probe`` 量出 044153 里有 16 格 0.65 m 的连贯水平面。决定性的下一步：

    这 16 格在现在的 ``nav_grid`` 里是 FREE 还是 OCC？
    FREE ⇒ 2D 没管住，净空过滤有真实价值
    OCC ⇒ 已经挡住了，功能是冗余的

## 对齐怎么做的（不手推）

不在 0.3 m 统计栅格上叠图——那套 ``acc_row = R + (ly-ay)`` 翻转代数已经栽过三次。
这里改成**直接在 mapper 自己的累加器网格上**重算高度直方图，形状和原点都用
``_acc_lo`` / ``res_m``，最后交给已经验证过的 ``mapper._cut_acc`` 换到 NavGrid 行序。
两条路子的输出 ``(h, w)`` 与 ``ng.g`` 逐格同位，交叉表不需要任何额外换算。

## 附带

``--dump`` 会把**看见过这些低面的关键帧**列出来（帧号、时刻、贡献点数、当时站位），
并画一张图：左 = nav 栅格叠上这批低面，右 = 贡献关键帧的轨迹。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import (KeyframeGridMapper, MapperConfig,          # noqa: E402
                                 _ground_correction)
from backend.nav_grid import FREE, OCC, UNK                                   # noqa: E402
from tools.precision_probe import load                                       # noqa: E402

HBIN = 0.05
MIN_PTS = 20        # 与 band_levels 一致：一格至少这么多票才谈"面"
PLANAR_TOL = 0.15   # 3×3 邻域众数高度极差
MIN_NB = 6
RECS = ("20261001_044153", "20260929_045615", "20261001_044120")


def band_pts(m: KeyframeGridMapper, c: MapperConfig, i: int):
    """第 i 关键帧落在 0.3~2.0 m 带、且在累加器范围内的 (ix, iy, h, w)。

    ``h`` 用的是 rasterize 之后**最终**的 cam_h——必须如此，否则和累加器口径不一致。
    """
    r = m._rot.get(i)
    if r is None or r[0] is None or not len(r[0]):
        return None
    rxy, rel, cnt, _R = r
    h = rel + np.float32(m.cam_h)
    h = h - np.float32(_ground_correction(h, cnt, rxy, c.ground_offset_max_m,
                                          c.ground_offset_min_range_m,
                                          c.ground_plane_max_deg, c.ground_plane_band_m))
    ok = (h > c.ground_tol_m) & (h <= c.obst_top_m)
    if not ok.any():
        return None
    xy = rxy[ok].astype(np.float64) + m._pose[i][:2, 3]
    ax, ay = m._acc_lo
    ix = np.floor(xy[:, 0] / c.res_m).astype(np.int64) - ax
    iy = np.floor(xy[:, 1] / c.res_m).astype(np.int64) - ay
    H, W = m._acc_g.shape
    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    cam = m._pose[i][:2, 3]
    rng = np.hypot(xy[:, 0] - cam[0], xy[:, 1] - cam[1])
    return ix[inside], iy[inside], h[ok][inside], cnt[ok][inside], rng[inside]


def low_height_grid(m: KeyframeGridMapper, c: MapperConfig, n_frames: int):
    """0.3~2.0 m 带的每格 (众数高度, 点数, 3×3 极差, 有效邻域数)，全在累加器行序。"""
    H, W = m._acc_g.shape
    lo = 12.0
    cache: dict[int, tuple] = {}
    for i in range(n_frames):
        r = band_pts(m, c, i)
        if r is not None:
            cache[i] = r
            lo = min(lo, float(r[2].min()))
    if not cache:
        return None
    nh = int((c.obst_top_m - lo) / HBIN) + 2
    hist = np.zeros((nh, H, W), np.float32)
    rsum = np.zeros((H, W), np.float32)
    for i, (ix, iy, h, w, rng) in cache.items():
        hb = np.floor((h - lo) / HBIN).astype(np.int64)
        good = (hb >= 0) & (hb < nh)
        if not good.any():
            continue
        flat = iy[good] * W + ix[good]
        hist += np.bincount(hb[good] * (H * W) + flat,
                            weights=w[good], minlength=nh * H * W).reshape(nh, H, W).astype(np.float32)
        rsum += np.bincount(flat, weights=w[good] * rng[good], minlength=H * W).reshape(H, W).astype(np.float32)
    n = hist.sum(axis=0)
    valid = n >= MIN_PTS
    if not valid.any():
        return None
    mode_h = np.where(valid, lo + hist.argmax(axis=0) * HBIN, np.nan)
    mean_r = np.where(n > 0, rsum / np.maximum(n, 1e-9), np.nan)
    spread, n_ok = _spread(mode_h, valid)
    planar = valid & (n_ok >= MIN_NB) & (spread <= PLANAR_TOL)
    return {"cache": cache, "lo": lo, "mode_h": mode_h, "n": n, "valid": valid,
            "spread": spread, "n_ok": n_ok, "planar": planar, "mean_r": mean_r,
            "H": H, "W": W}


def _spread(mode_h: np.ndarray, valid: np.ndarray):
    M = np.where(valid, mode_h, np.nan)
    H, W = M.shape
    stack = np.full((9, H, W), np.nan)
    for k, (dy, dx) in enumerate([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]):
        ys0, ys1 = max(0, dy), H + min(0, dy)
        xs0, xs1 = max(0, dx), W + min(0, dx)
        stack[k, ys0:ys1, xs0:xs1] = M[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
    with np.errstate(invalid="ignore"):
        return np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0), np.sum(~np.isnan(stack), axis=0)


class SpyMapper(KeyframeGridMapper):
    """截下 ``_by_quality`` 的入参与出参——不复算，直接看生产路径自己拿到了什么。

    这些数组是 **rasterize 的栅格行序**（行 0 = 最小 y），还没做末尾那一次 ``g[::-1]``。
    """

    def _by_quality(self, base_occ, n_g, n_o, o_n, o_m, g_n, o_mk):
        occ, restored, rejected = super()._by_quality(base_occ, n_g, n_o, o_n, o_m, g_n, o_mk)
        self.spy = {"base_occ": base_occ.copy(), "n_g": n_g.copy(), "n_o": n_o.copy(),
                    "o_n": o_n.copy(), "o_m": o_m.copy(), "g_n": g_n.copy(),
                    "occ": occ.copy(), "restored": restored.copy(), "rejected": rejected.copy()}
        return occ, restored, rejected


def probe(rec_name: str, dump: bool):
    frames = load(ROOT / "navmesh_recordings" / rec_name)
    if not frames:
        print(f"\n=== {rec_name} ===\n  没有关键帧")
        return None
    c = MapperConfig()
    m = SpyMapper(c)
    m._rec_name = rec_name
    for i, fr in enumerate(frames):
        m.add_keyframe(i, fr[1], fr[2])
    ng = m.rasterize()                     # 这一步定下 _acc_lo / _grid_lo / _last_grid
    ng.build(radius_m=0.25, walked=m.walked())
    s = ng.meta.world_scale

    print(f"\n=== {rec_name} ===  {len(frames)} 关键帧  栅格 {ng.grid.shape}  "
          f"res {c.res_m} m  world_scale {s}")
    if abs(s - 1.0) > 1e-9:
        print(f"  （world_scale = {s}）")
    acc = low_height_grid(m, c, len(frames))
    if acc is None:
        print("  0.3~2.0 m 带里没有一格攒够点数")
        return None

    # ---- 累加器行序 → NavGrid 行序，走 _cut_acc，不手推 ----
    mode_h = np.nan_to_num(acc["mode_h"], nan=0.0)
    g_mode = m._cut_acc(mode_h)
    g_valid = m._cut_acc(acc["valid"].astype(np.float32)) > 0.5
    g_planar = m._cut_acc(acc["planar"].astype(np.float32)) > 0.5
    g_n = m._cut_acc(acc["n"])
    if g_mode.shape != ng.grid.shape:
        print(f"  !! 形状对不上：_cut_acc 给 {g_mode.shape}，nav 栅格 {ng.grid.shape}")
        return None

    sel = np.flatnonzero(g_planar.ravel())
    if not len(sel):
        print("  没有低水平面 ⇒ 净空过滤无对象")
        return None
    print(f"  低水平面 {len(sel)} 格（累加器 {int(acc['planar'].sum())} 格）"
          f"  众数高度 {g_mode[g_planar].min():.2f}..{g_mode[g_planar].max():.2f} m")

    kind = np.where(ng.grid.ravel()[sel] == OCC, "OCC",
                    np.where(ng.grid.ravel()[sel] == FREE, "FREE", "UNK"))
    center = ng.center.ravel()[sel]
    clr = ng.clearance.ravel()[sel]
    print("\n  这些格在现在的 nav 栅格里：")
    for k in ("OCC", "FREE", "UNK"):
        n = int((kind == k).sum())
        if n:
            print(f"    {k:5s} {n:4d} 格 ({n / len(sel) * 100:5.1f}%)")
    n_center = int(center.sum())
    print(f"  其中算作可走中心格（净空 ≥ 0.25 m）{n_center} / {len(sel)}")
    if n_center:
        print(f"    可走格的净空 min {clr[center].min():.2f} 中位 {np.median(clr[center]):.2f} m")
    low_free = int(((kind == "FREE") & center).sum())
    if low_free == 0:
        print("  ⇒ 结论：这些低面**全部**已经被 2D 栅格当障碍挡住了，净空过滤是冗余的")
    else:
        print(f"  ⇒ 结论：**{low_free} 格**现在被标为可走，净空过滤能真正拦下它们")

    # ---- 高度分层：0.3~0.5 m 贴着 ground_tol，多半是地面修正残余，不是"低面" ----
    hh = g_mode[g_planar]
    idx = np.flatnonzero(g_planar.ravel())
    ctr = ng.center.ravel()
    edges = [0.3, 0.5, 0.8, 1.2, 1.6, 2.01]
    print("\n  按众数高度分档（0.5 m 以下贴着 ground_tol，多半是地面修正残余）：")
    for a, b in zip(edges[:-1], edges[1:]):
        k = (hh >= a) & (hh < b)
        if not k.any():
            continue
        walk_n = int(ctr[idx[k]].sum())
        print(f"    {a:.2f}~{b:.2f} m  {int(k.sum()):5d} 格  其中可走 {walk_n:5d} "
              f"({walk_n / max(int(k.sum()), 1) * 100:5.1f}%)")

    # ---- 归因：可走的低面，是旧规则本来就没判障碍，还是被质量分层放行的 ----
    spy = {k: v[::-1] if v.ndim == 2 else v for k, v in m.spy.items()}
    wsel = idx[ctr[idx]]
    print(f"\n  归因（只看 {len(wsel)} 个可走的低面格）——")
    if not len(wsel):
        print("    没有可走的低面格")
    else:
        # 我自己重算的带内点数 vs mapper 记的 n_o：两者应该相等，不等就说明有人动过 n_o
        mine = m._cut_acc(acc["n"]).ravel()[wsel]
        no = spy["n_o"].ravel()[wsel]
        ngs = spy["n_g"].ravel()[wsel]
        bo = spy["base_occ"].ravel()[wsel] > 0
        rs = spy["restored"].ravel()[wsel]
        print(f"    ray_clear={c.ray_clear}  本次被看穿清零的格数 {m.ray_cleared_cells}")
        print(f"    我重算的带内点数  中位 {np.median(mine):.1f}   "
              f"mapper 的 n_o 中位 {np.median(no):.1f}   （两者不等 ⇒ n_o 被改过）")
        print(f"    n_o 被清成 0 的：{int((no == 0).sum())} / {len(wsel)} 格"
              f"（我这边点数 >0 的 {int((mine > 0).sum())}）")
        print(f"    扁平规则（n_o>min_pts 且 n_o≥{c.occ_ground_ratio}·n_g，3×3 多数）判为障碍的 "
              f"{int(bo.sum())} / {len(wsel)}")
        print(f"    其中被质量分层降级回 FREE 的 {int((bo & rs).sum())} 格")
        print(f"    根本没判障碍的 {int((~bo).sum())} 格  ← 旧规则的老漏")
        print(f"      这批 n_g 中位 {np.median(ngs):.1f}  n_o 中位 {np.median(no):.1f}  "
              f"（min_pts {c.min_pts}，地面压制需 n_o ≥ {c.occ_ground_ratio}·n_g"
              f" ≈ {c.occ_ground_ratio * np.median(ngs):.1f}）")

    # ---- 判真假：观测距离。δz = z²/25.5 m，1.5 m→9 cm、3 m→35 cm、4 m→61 cm ----
    g_r = m._cut_acc(np.nan_to_num(acc["mean_r"], nan=0.0)).ravel()[idx]
    g_h = g_mode.ravel()[idx]
    ctr_all = ctr[idx]
    occ_kind = ng.grid.ravel()[idx] == OCC
    print("\n  观测距离（加权平均，看穿判据的可信度随距离塌掉：δz=z²/25.5 m）")
    for name, k in (("可走的低面格", ctr_all), ("已判 OCC 的低面格", ~ctr_all & occ_kind),
                    ("FREE/UNK 的低面格", ~ctr_all & ~occ_kind)):
        if not k.any():
            continue
        r = g_r[k]
        print(f"    {name:18s} n={int(k.sum()):5d}  距离中位 {np.median(r):5.2f} m  "
              f"p25 {np.percentile(r, 25):5.2f}  p75 {np.percentile(r, 75):5.2f}  "
              f"高度中位 {np.median(g_h[k]):.2f} m")
    near_real = ctr_all & (g_r < 2.0) & (g_h > 0.5)
    print(f"\n  ⇒ 可走 ∧ 近距(<2 m) ∧ 高度>0.5 m 的：{int(near_real.sum())} 格"
          f"  —— 这批躲不开深度误差，最可能是真物件")
    if near_real.any():
        print(f"      高度中位 {np.median(g_h[near_real]):.2f} m  "
              f"其中 OCC 态 {int(occ_kind[near_real].sum())} 格")
    far_sc = ctr_all & (g_r >= 3.0)
    print(f"  ⇒ 可走 ∧ 远距(≥3 m) 的：{int(far_sc.sum())} 格"
          f"  —— 这批 δz≥0.35 m，抖上 1 m 完全正常，很可能是地面散点")

    # ---- 这 496 格是一件家具还是一片噪声：高度集中度 + 空间紧凑度 ----
    vh = g_h[near_real]
    print(f"\n  近距那 {len(vh)} 格的高度分布（家具 ⇒ 集中在台面高；散点 ⇒ 摊开）：")
    edges2 = [0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.01]
    for a, b in zip(edges2[:-1], edges2[1:]):
        k = int(((vh >= a) & (vh < b)).sum())
        if k:
            bar = "█" * max(1, k * 40 // max(len(vh), 1))
            print(f"    {a:.2f}~{b:.2f} m  {k:5d} 格 {bar}")
    print(f"    高度 四分位 {np.percentile(vh, 25):.2f} / {np.median(vh):.2f} / "
          f"{np.percentile(vh, 75):.2f} m   极差 {np.percentile(vh, 95) - np.percentile(vh, 5):.2f} m")
    rr, ccn = np.divmod(idx[near_real], ng.grid.shape[1])
    ncomp = _n_components(rr, ccn, 0.10)
    print(f"    空间上（0.10 m 容差）聚成 {ncomp} 团  最大团 {max(_label_sizes(rr, ccn, 0.10))} 格")

    dump_frames(m, c, ng, sel, acc, keep=near_real, idx=idx)
    veto_profile(m, c, ng, idx, near_real, acc)
    save_art(ng, idx, g_h, g_r, ctr_all, near_real, occ_kind, rec_name)

    if dump:
        pass
    return {"sel": sel, "g_planar": g_planar, "kind": kind, "low_free": low_free,
            "rec": rec_name, "n": len(sel)}


def _label_sizes(rows, cols, tol):
    """把 (row, col) 按 tol（米 → 格）做并查集，返回各团大小。"""
    res = _cc(rows.astype(np.int64), cols.astype(np.int64), int(round(tol / 0.1)))
    _, cnt = np.unique(res[res > 0], return_counts=True)
    return sorted(cnt.tolist(), reverse=True)


def _n_components(rows, cols, tol):
    return len(_label_sizes(rows, cols, tol))


def _cc(rows: np.ndarray, cols: np.ndarray, r_tol: int) -> np.ndarray:
    """4-邻接 + 行/列容差 r_tol 的连通分量标号（BFS，纯 numpy/python 都行，格数少）。"""
    key = rows * 100000 + cols
    order = np.argsort(key)
    key_s, rows_s, cols_s = key[order], rows[order], cols[order]
    pos = {int(k): i for i, k in enumerate(key_s)}
    lab = np.full(len(key_s), -1, np.int64)
    comp = 0
    for i in range(len(key_s)):
        if lab[i] >= 0:
            continue
        stack = [i]
        lab[i] = comp
        while stack:
            j = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = int(rows_s[j]) + dr, int(cols_s[j]) + dc
                for aa in range(a - r_tol, a + r_tol + 1):
                    for bb in (b, b + dc * r_tol):
                        t = pos.get(aa * 100000 + bb)
                        if t is not None and lab[t] < 0:
                            lab[t] = comp
                            stack.append(t)
        comp += 1
    out = np.empty(len(key), np.int64)
    out[order] = lab
    return out


def nav_to_acc(ng, m: KeyframeGridMapper, rows: np.ndarray, cols: np.ndarray):
    """NavGrid (row, col) → 累加器 (iy, ix)，直接照抄 ``_cut_acc`` 自己的换算，不走世界坐标。

    ``_cut_acc`` 里 ``out[r0+j] = sub[s_hi-1-j]`` ⇒ ``acc_row = h-1+dh-nav_row``；
    ``out[c_lo-dw+jj] = sub[c_lo+jj]`` ⇒ ``acc_col = nav_col + dw``。
    """
    h = ng.grid.shape[0]
    (ax, ay) = m._acc_lo
    (lx, ly) = m._grid_lo
    dh, dw = ly - ay, lx - ax
    return h - 1 + dh - rows, cols + dw


def veto_profile(m, c, ng, idx, near_real, acc):
    """被清零的格，逐 z 层的 (打中, 看穿) 剖面——验证"射线掠过墙顶"这条机理。

    层号 iz 对应高度 ``ground_tol + (iz+0.5)·ray_z_m``。命中只可能落在物体的**实心**层上；
    物体上方的层是空体素，只可能吃 ``miss``。所以判据的关键是：**有打中的那几层，
    是不是也被掠过的射线冲掉了**。
    """
    if m._ray_hit is None:
        print("\n  （ray_clear 关着，没有 _ray_hit）")
        return
    hit = np.stack([m._cut_acc(m._ray_hit[z]) for z in range(m._ray_hit.shape[0])])
    mis = np.stack([m._cut_acc(m._ray_mis[z]) for z in range(m._ray_mis.shape[0])])
    H = hit.reshape(len(hit), -1)[..., idx]
    M = mis.reshape(len(mis), -1)[..., idx]
    nlay = len(hit)
    tol = c.ground_tol_m + (np.arange(nlay) + 0.5) * c.ray_z_m
    print(f"\n  逐层剖面（{int(near_real.sum())} 个近距低面格；层高对应高度见下）"
          f"  打中总计 中位 {np.median(H[:, near_real].sum(axis=0)):.0f}"
          f"  看穿总计 中位 {np.median(M[:, near_real].sum(axis=0)):.0f}")
    print(f"    {'层':>3s} {'高度m':>7s} {'有打中的格':>10s} {'其中看穿≥打中':>14s} {'该层看穿总计':>12s}")
    for z in range(nlay):
        hz = H[z][near_real]
        mz = M[z][near_real]
        has = hz > 0
        if not has.any() and mz.sum() == 0:
            continue
        bad = has & (mz >= c.ray_beta * hz)
        print(f"    {z:3d} {tol[z]:7.2f} {int(has.sum()):10d} {int(bad.sum()):14d} "
              f"{int(mz.sum()):12d}")
    any_pass = ((H[:, near_real] > 0) & (M[:, near_real] < c.ray_beta * H[:, near_real])).any(axis=0)
    print(f"    ⇒ 至少有一层通过判据 ⇒ 不该被清：{int(any_pass.sum())} 格"
          f"   实测被清：{int(near_real.sum()) - int(any_pass.sum())} 格")


def dump_frames(m, c, ng, sel, acc, keep=None, idx=None):
    """哪些关键帧看见了这批低面。``keep``：压缩后的布尔掩码（长度 = len(sel)）。落盘成 JSON。"""
    import json
    w = ng.grid.shape[1]
    rows, cols = np.divmod(sel, w)
    iy, ix = nav_to_acc(ng, m, rows, cols)
    ok = ((iy >= 0) & (iy < acc["H"]) & (ix >= 0) & (ix < acc["W"]))
    if keep is not None:
        ok = ok & keep                      # keep 与 iy/ix 同长（压缩后的掩码）
    if not ok.any():
        print("  反算回累加器后没有落在范围内的格")
        return [], {}
    want = iy[ok] * acc["W"] + ix[ok]
    hits: dict[int, int] = {}
    poses: dict[int, tuple] = {}
    for i, (jx, jy, _hh, _ww, _rg) in acc["cache"].items():
        n = int(np.isin(jy * acc["W"] + jx, want).sum())
        if n:
            hits[i] = n
            poses[i] = tuple(m._pose[i][:2, 3])
    ks = sorted(hits, key=lambda i: -hits[i])
    print(f"\n  看见过它们的��键帧 {len(ks)} / {len(acc['cache'])} 有效帧")
    print("    帧号   贡献点   站位 (x, y) m")
    for i in ks[:20]:
        x, y = poses[i]
        print(f"    {i:5d}  {hits[i]:7d}   ({x:7.2f}, {y:7.2f})")
    if len(ks) > 20:
        print(f"    …另有 {len(ks) - 20} 帧")
    out = ROOT / "tools" / f"low_ceiling_kf_{rec_name_of(m)}.json"
    payload = {"frames": [{"i": int(i), "pts": hits[i], "xy_m": [round(poses[i][0], 3),
                                                               round(poses[i][1], 3)]} for i in ks]}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"    全部 {len(ks)} 帧已落盘 {out.relative_to(ROOT)}")
    return ks, poses


def rec_name_of(m):
    return getattr(m, "_rec_name", "rec")


def save_art(ng, idx, g_h, g_r, ctr_all, near_real, occ_kind, rec_name):
    """左：nav 栅格 + 这批近距低面；右：高度 vs 观测距离。"""
    from tools.plotfont import use_cjk
    use_cjk()
    import matplotlib.pyplot as plt
    g = ng.grid
    rgb = np.stack([g, g, g], -1).astype(np.uint8)
    h, w = g.shape
    flat = rgb.reshape(-1, 3)
    flat[idx] = np.array([70, 70, 70], np.uint8)
    flat[idx[occ_kind]] = np.array([40, 90, 200], np.uint8)
    flat[idx[near_real]] = np.array([220, 40, 40], np.uint8)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(flat.reshape(h, w, 3), interpolation="nearest")
    ax[0].set_title(f"{rec_name}  nav 栅格\n红=近距低面且可走 {int(near_real.sum())} 格   "
                    f"蓝=低面但已判 OCC   深灰=其他低面")
    ax[0].set_xlabel("栅格列（x）"); ax[0].set_ylabel("栅格行（y，行 0 = 最大 y）")
    ax[1].scatter(g_r[~near_real], g_h[~near_real], s=4, c="0.7",
                  label=f"其他低面 {int((~near_real).sum())}")
    ax[1].scatter(g_r[near_real], g_h[near_real], s=6, c="crimson",
                  label=f"近距+可走+>0.5 m  {int(near_real.sum())}")
    ax[1].axvline(2.0, ls="--", c="k", lw=1)
    ax[1].axvline(3.0, ls=":", c="k", lw=1)
    ax[1].set_xlabel("观测距离 m（δz = z^2/25.5）"); ax[1].set_ylabel("低面众数高度 m")
    ax[1].set_ylim(0, 2.05); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    out = ROOT / "tools" / f"low_ceiling_{rec_name}.png"
    fig.savefig(out, dpi=110)
    print(f"\n  图已存 {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recs", nargs="*", default=None)
    ap.add_argument("--dump", action="store_true", help="列出看见过低面的关键帧")
    a = ap.parse_args()
    tot = {"sel": 0, "low_free": 0}
    for r in (a.recs or list(RECS)):
        res = probe(r, a.dump)
        if res:
            tot["sel"] += res["n"]
            tot["low_free"] += res["low_free"]
    if tot["sel"]:
        print(f"\n合计：低水平面 {tot['sel']} 格，其中被标为可走 {tot['low_free']} 格")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
