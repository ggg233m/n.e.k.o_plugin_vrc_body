# -*- coding: utf-8 -*-
"""把"逐帧深度噪声"和"多视角错位"分开量。

同一段录制、同一个判定规则，只改**累积多少个关键帧**：

  * 单帧地图（只喂 1 个关键帧）里完全没有位姿误差——所有点都来自同一个坐标系。
    它的"观测地面被判成障碍"的比例 = **逐帧深度噪声**的天花板。
  * 累积 N 帧后，同一个物理位置被 N 个坐标系投票。如果位姿准，比例应该随 N **持平或下降**
    （多视角投票互相抵消噪声）；如果位姿在漂，比例会**随 N 上涨**——同一个格子一会儿被说成
    地面一会儿被说成障碍，票就堆在两个格子之间的缝里。

所以"随 N 涨"就是位姿精度问题，"与 N 无关"才是深度精度问题。

    python -m tools.precision_probe 20261001_044153
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_grid import FREE, OCC, UNK          # noqa: E402
from backend.nav_mapping import KeyframeGridMapper, MapperConfig  # noqa: E402

STEPS = (1, 2, 5, 10, 25, 50, 100, 200, 0)          # 0 = 全部
REC = "20261001_044153"


def load(rec: Path):
    ev = [json.loads(x) for x in
          (rec / "events.jsonl").read_text(encoding="utf-8").strip().splitlines() if x.strip()]
    fp = rec / "final_poses.npz"
    final: dict[int, np.ndarray] = {}
    if fp.exists():
        z = np.load(fp)
        final = {int(i): np.asarray(T, np.float64) for i, T in zip(z["ids"], z["T"])}
    frames = []
    cache: dict[int, dict] = {}
    for e in ev:
        if e.get("kind") != "kf":
            continue
        k = int(e["k"])
        f = rec / "kf" / f"{k:06d}.npz"
        if not f.exists():
            continue
        if k not in cache:
            cache.clear()
            cache[k] = np.load(f)
        if e.get("refresh"):
            continue                      # refresh 帧会让计数对不齐，这里只用常态帧
        # raw = 记录时写进 kf 的位姿（未回环修正）；pose = 有 final_poses 就用回环修正后的
        raw = np.asarray(cache[k]["T_map"], np.float64)
        frames.append((k, cache[k]["pts"].astype(np.float32), final.get(k, raw), raw))
    return frames


def floor_polluted(ng, s: float) -> tuple[int, int]:
    """（被判成障碍的格, 观测到地面的格）——只统计有地面观测的格。"""
    g = ng.grid
    seen_ground = g != UNK
    occ = (g == OCC) & seen_ground
    return int(occ.sum()), int(seen_ground.sum())


def corridor_blockers(ng, trail: list[np.ndarray], half_m: float = 0.5) -> int:
    """走廊上还有几个障碍格（与 tools/q_tier_ab.py 同口径）。"""
    import cv2
    H, W = ng.grid.shape
    m = np.zeros((H, W), np.uint8)
    ox, oy = ng.meta.origin_xy_m
    r = ng.meta.resolution_m
    for p in trail:
        cx, cy = int((p[0] - ox) / r), int((p[1] - oy) / r)
        if 0 <= cx < H and 0 <= cy < W:
            cv2.circle(m, (cy, cx), max(1, int(round(half_m / r))), 1, -1)
    return int(((m > 0) & (ng.grid == OCC)).sum())


def pose_ab(rec: Path, frames) -> None:
    """同一批关键帧，只换位姿：回环修正后的 final_poses vs 记录时的 kf T_map。

    这是唯一能把"位姿精度"从其它一切里摘出来的对照——关键帧、点云、判定规则全都一样，
    唯一的差别是位姿差了 0.18~1.09 m（044153 的回环修正量）。地图差得多 ⇒ 位姿是瓶颈。
    """
    fp = rec / "final_poses.npz"
    if not fp.exists():
        print(f"\n[{rec.name}] 没有 final_poses.npz，跳过位姿对照（这段本来就没有回环修正）")
        return
    z = np.load(fp)
    final = {int(i): np.asarray(T, np.float64) for i, T in zip(z["ids"], z["T"])}
    drift = []
    for k, _pts, T1, T0 in frames:
        drift.append(float(np.linalg.norm(T1[:2, 3] - T0[:2, 3])))
    drift = np.asarray(drift)
    print(f"\n[{rec.name}] 回环修正量 平均 {drift.mean():.3f} / p95 {np.percentile(drift, 95):.3f} "
          f"/ 最大 {drift.max():.3f} m")
    print(f"{'位姿来源':>16} {'障碍格':>8} {'走廊障碍':>8} {'有地面观测格':>12} {'地面被污染%':>12}")
    for tag, which in (("回环修正后", 2), ("记录时 T_map", 3)):
        m = KeyframeGridMapper(MapperConfig(q_tiers=False))
        trail = []
        for i, fr in enumerate(frames):
            T = fr[which]
            m.add_keyframe(i, fr[1], T)
            trail.append(T[:2, 3].copy())
        ng = m.rasterize()
        o, f = floor_polluted(ng, ng.meta.world_scale)
        cb = corridor_blockers(ng, trail)
        print(f"{tag:>16} {o:>8} {cb:>8} {f:>12} {100.0 * o / max(f, 1):>11.2f}%")


def closest_range(frames, cfg: MapperConfig, mapper):
    """给每个格记下"历史上离它最近的观测距离"（追踪米），返回**累加器布局**的稠密数组。

    之前用打包键 + divmod 解回来，错了三次（减重偏移、x/y 互换）。这里直接用累加器坐标
    做稠密数组，完全不打包——没有解码就没有解码的 bug。
    """
    from backend.nav_mapping import _ground_correction, _voxelize
    (ax, ay), (H, W) = mapper._acc_lo, mapper._acc_g.shape
    out = np.full(H * W, np.inf)
    cam_h = mapper.cam_h
    for fr in frames:
        pts, T = fr[1], fr[2]
        p = pts[np.hypot(pts[:, 0], pts[:, 1]) <= cfg.range_m]
        if not len(p):
            continue
        cen, w = _voxelize(p, cfg.vox_xy_m, cfg.vox_z_m)
        w = w.astype(np.float64)
        q = cen @ T[:3, :3].T.astype(np.float32)
        rxy, rel = q[:, :2].copy(), q[:, 2].copy()
        h = rel + np.float32(cam_h)
        h = h - np.float32(_ground_correction(h, w, rxy, cfg.ground_offset_max_m,
                                              cfg.ground_offset_min_range_m,
                                              cfg.ground_plane_max_deg, cfg.ground_plane_band_m))
        sel = (np.abs(h) <= cfg.ground_tol_m) | ((h > cfg.ground_tol_m) & (h < cfg.obst_top_m))
        if not sel.any():
            continue
        rxy = rxy[sel].astype(np.float64)
        d = np.hypot(rxy[:, 0], rxy[:, 1])
        xy = rxy + T[:2, 3]
        ix = np.floor(xy[:, 0] / cfg.res_m).astype(np.int64)
        iy = np.floor(xy[:, 1] / cfg.res_m).astype(np.int64)
        flat = (iy - ay) * W + (ix - ax)
        keep = (flat >= 0) & (flat < H * W)
        flat, d = flat[keep], d[keep]
        order = np.argsort(flat)
        flat, d = flat[order], d[order]
        uniq, start = np.unique(flat, return_index=True)
        dmin = np.minimum.reduceat(d, start)      # 本帧每格唯一，可直接散写
        cur = out[uniq]
        out[uniq] = np.where(dmin < cur, dmin, cur)
    return out.reshape(H, W)


def range_breakdown(frames, cfg: MapperConfig) -> None:
    """把地图按"该格最近一次被观测时的距离"分档，看假障碍率怎么随距离走。"""
    m = KeyframeGridMapper(cfg)
    for i, fr in enumerate(frames):
        m.add_keyframe(i, fr[1], fr[2])
    ng = m.rasterize()
    near = closest_range(frames, cfg, m)          # 累加器布局 (H, W)
    H, W = near.shape
    h, w = ng.grid.shape
    x0, y0 = m._acc_lo
    lx, ly = m._grid_lo
    # rasterize 的切片式 n_g[R] = acc[R + (ly - y0)]，再 g[::-1] ⇒ NavGrid 行 h-1-R。
    # 符号先前写成 −，方向反了（整体平移、不报错，但分档边界不可信 —— 已修）。
    dh, dw = ly - y0, lx - x0
    s_lo, s_hi = max(0, dh), min(H, h + dh)          # 未翻转行 R ⇒ 累加器行 R + dh
    c_lo, c_hi = max(0, dw), min(W, w + dw)
    sub = near[s_lo:s_hi, c_lo:c_hi]
    # R = s - dh，s ∈ [s_lo, s_hi) ⇒ R ∈ [0, h)；翻转后 sub[j] ↔ NavGrid 行 j（已对齐）
    sub = sub[::-1]
    g_row0 = 0
    g_col0 = c_lo - dw                          # 列不翻转，起点是 c_lo - dw
    obs = np.isfinite(sub) & (ng.grid[g_row0:g_row0 + sub.shape[0],                                     g_col0:g_col0 + sub.shape[1]] != UNK)
    occ = obs & (ng.grid[g_row0:g_row0 + sub.shape[0], g_col0:g_col0 + sub.shape[1]] == OCC)
    print(f"\n[按最近观测距离分档] cam_h={m.cam_h:.3f}   落进图的格 {int(obs.sum())}")
    print(f"{'最近观测距离':>14} {'格数':>8} {'其中障碍格':>10} {'障碍率':>8}")
    for lo_, hi_ in ((0.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 99.0)):
        msk = obs & (sub >= lo_) & (sub < hi_)
        n = int(msk.sum())
        if not n:
            continue
        o = int((msk & occ).sum())
        tag = f"{lo_}-{hi_} m" if hi_ < 99.0 else f"> {lo_} m"
        print(f"{tag:>14} {n:>8} {o:>10} {100.0 * o / n:>7.2f}%")


def main() -> None:
    rec_name = sys.argv[1] if len(sys.argv) > 1 else REC
    rec = ROOT / "navmesh_recordings" / rec_name
    frames = load(rec)
    N = len(frames)
    print(f"录制 {rec_name}: {N} 关键帧")
    print("沿整条轨迹**均匀取样** N 帧：覆盖面积基本不变，只有冗余在变。")
    print("  地面被污染% 随 N 涨 ⇒ 多视角错位（位姿）；与 N 无关 ⇒ 逐帧深度噪声。\n")
    print(f"{'取样帧数':>8} {'障碍格':>8} {'有地面观测格':>12} {'地面被污染%':>12}")
    for n in STEPS:
        m = N if n == 0 else n
        idx = np.linspace(0, N - 1, m).round().astype(int)
        mp = KeyframeGridMapper(MapperConfig(q_tiers=False))     # 扁平口径，量的是"原始"行为
        for j, i in enumerate(idx):
            fr = frames[i]
            mp.add_keyframe(j, fr[1], fr[2])
        ng = mp.rasterize()
        o, f = floor_polluted(ng, ng.meta.world_scale)
        print(f"{len(idx):>8} {o:>8} {f:>12} {100.0 * o / max(f, 1):>11.2f}%")
    pose_ab(rec, frames)
    for tag, cfg in (("扁平计数（改动前）", MapperConfig(q_tiers=False)),
                     ("质量分层（改动后）", MapperConfig(q_tiers=True))):
        print(f"\n=========== {tag} ===========")
        range_breakdown(frames, cfg)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()

