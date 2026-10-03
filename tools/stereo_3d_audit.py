# -*- coding: utf-8 -*-
"""把审计的"精确 3D 墙体已证伪"结论，拿**双目路线**重新量一遍。

    python -m tools.stereo_3d_audit [录制名]

## 为什么值得重做

`research/audit/incr_geom_falsified_3d_walls/`（2026-09-21）判"精确 3D 结案"，但它比的是
(a) 单目深度模型 DA-V2 + 位姿、(b) 图像特征三角化 + 位姿，两条**共用同一套位姿**。而：

1. **它用的是未被回环优化污染的 DR 位姿**（MANIFEST §输入明写"取 x0,z0,yaw（DR，未被回环
   优化污染）"）。系统现在有回环修正了（044153：平均 0.18 / 最大 1.09 m），当时没测过。
2. **深度路线是单目**（DA-V2，448×252）。它有明确的远场偏差（6–9 m 比值 0.62、9–12 m 0.46）。
   在线系统实际用的是**双目**，不吃这个偏差。

## 指标（与审计逐字一致，见 _incr_geom.py:128 local_plan_med）

    对每个点取 0.40 m 邻域 → 最小二乘平面 → 记 σ_min / sqrt(邻居数)（= 面外 RMS 距离）
    取全体中位数。单位追踪米。止损线 0.07 m。

## 关键分组

把跨帧一致性单独拎出来——这是审计判定"瓶颈在位姿"的依据：

    单关键帧      位姿误差 = 0（所有点同一个坐标系）⇒ 纯深度噪声下限
    相邻两关键帧  位姿误差 ≈ 1 s 内的漂移
    全部 N 关键帧 位姿误差 = 全程漂移 + 回环修正
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.nav_grid import OCC                       # noqa: E402
from backend.nav_mapping import KeyframeGridMapper, MapperConfig  # noqa: E402

REC = "20261001_044153"
RADIUS = 0.40
MINN = 6
N_PTS = 20000          # KD 树 + Python 循环，按审计同样口径降采样
STOP_LOSS_M = 0.07


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
            continue
        raw = np.asarray(cache[k]["T_map"], np.float64)
        frames.append((k, cache[k]["pts"].astype(np.float32), final.get(k, raw), raw))
    return frames


def local_plan(P: np.ndarray, radius: float = RADIUS, minn: int = MINN) -> tuple[float, float, int]:
    """与 _incr_geom.py:128 逐字同口径：0.40 m 邻域最小奇异值 / sqrt(邻居数)，取中位。"""
    P = np.asarray(P, np.float64)
    if P.shape[0] < minn:
        return float("nan"), float("nan"), 0
    if P.shape[0] > N_PTS:
        rs = np.random.default_rng(2)                     # 审计用 seed=2
        P = P[rs.choice(P.shape[0], N_PTS, replace=False)]
    nb = cKDTree(P).query_ball_point(P, radius, workers=-1)
    res = np.full(len(P), np.nan)
    for i, nl in enumerate(nb):
        if len(nl) < minn:
            continue
        A = P[nl] - P[nl].mean(0)
        sv = np.linalg.svd(A, compute_uv=False)
        res[i] = sv[-1] / math.sqrt(len(nl))
    ok = np.isfinite(res)
    if not ok.any():
        return float("nan"), float("nan"), 0
    return float(np.median(res[ok])), float(np.percentile(res[ok], 90)), int(ok.sum())


def clouds(frames, mapper: KeyframeGridMapper, which: int, wall_mask=None):
    """把所有关键帧点投到地图系；which=2 回环修正后 / 3 记录时。可选只保留墙点。"""
    out = []
    for fr in frames:
        pts, T = fr[1], fr[which]
        p = pts[np.hypot(pts[:, 0], pts[:, 1]) <= mapper.cfg.range_m]
        if not len(p):
            continue
        out.append(p.astype(np.float64) @ T[:3, :3].T + T[:3, 3])
    P = np.vstack(out) if out else np.zeros((0, 3))
    return P if wall_mask is None else P[wall_mask(P)]


def wall_selector(ng, mapper: KeyframeGridMapper):
    """挑"真墙"点：落在最终障碍格（分层规则确认过的）里的点。

    映射关系照 rasterize 的切片式抄：``n_g[oy0-ly : oy1-ly] = acc[oy0-y0 : oy1-y0]``
    ⇒ **未翻转行 R = 累加器行 s - (ly - y0)**，列 C = 累加器列 c - (lx - x0)；
    再 ``g[::-1]`` ⇒ NavGrid 行 h-1-R。（这个符号先前写成 +，会整体平移 2×dh 不报错，
    分档边界就不对了。）
    远场地板票已被分层否证成 unknown，所以这里选出的是被近/中距佐证过的表面。
    """
    lx, ly = mapper._grid_lo
    (ax, ay) = mapper._acc_lo
    h, w = ng.grid.shape
    dh, dw = ly - ay, lx - ax
    R = np.arange(h) + dh                 # NavGrid 未翻转行 R ↔ 累加器行 R + dh
    C = np.arange(w) + dw
    occ_r = np.zeros(h, bool)
    occ_c = np.zeros(w, bool)
    ok_r = (R >= 0) & (R < mapper._acc_g.shape[0])
    ok_c = (C >= 0) & (C < mapper._acc_g.shape[1])
    occ_r[ok_r] = (mapper._acc_o[R[ok_r], :] > 0.5).any(axis=1)
    occ_c[ok_c] = (mapper._acc_o[:, C[ok_c]] > 0.5).any(axis=0)
    # 只保留 ng 里真的是障碍格的那部分
    cell_occ = occ_r[:, None] & occ_c[None, :] & (ng.grid == OCC)
    a_r0, a_c0 = -dh, -dw                 # NavGrid(未翻转) 行/列 → 累加器下标
    Rg = h - 1 - np.arange(h)             # ng 行（翻转后）→ 未翻转行

    def sel(P: np.ndarray) -> np.ndarray:
        cs = np.floor(P[:, 0] / mapper.cfg.res_m).astype(np.int64) - ax
        s = np.floor(P[:, 1] / mapper.cfg.res_m).astype(np.int64) - ay
        r = h - 1 - (s + dh)              # ng 行
        c = cs + dw
        ok = (0 <= r) & (r < h) & (0 <= c) & (c < w)
        out = np.zeros(len(P), bool)
        out[ok] = cell_occ[r[ok], c[ok]]
        return out
    return sel, int(cell_occ.sum())


def main() -> None:
    rec_name = sys.argv[1] if len(sys.argv) > 1 else REC
    rec = ROOT / "navmesh_recordings" / rec_name
    frames = load(rec)
    if not frames:
        print(f"{rec_name}: 没有可用关键帧")
        return
    m = KeyframeGridMapper(MapperConfig(q_tiers=True))
    for i, fr in enumerate(frames):
        m.add_keyframe(i, fr[1], fr[2])
    ng = m.rasterize()
    sel, n_wall_cell = wall_selector(ng, m)
    print(f"录制 {rec_name}: {len(frames)} 关键帧，障碍格 {int((ng.grid == OCC).sum())}（墙点来源）")
    print(f"指标：0.40 m 邻域最小奇异值/sqrt(n)，中位值，单位追踪米；止损线 {STOP_LOSS_M} m\n")

    print(f"{'分组':>26} {'点数':>8} {'平面性中位':>10} {'p90':>8}   判定")
    for which, tag in ((3, "记录时 DR 位姿"), (2, "回环修正后位姿")):
        # 单关键帧：刚性变换不改变平面性，所以直接用 base 系（此时位姿误差恒为 0）
        b = frames[0][1]
        b = b[np.hypot(b[:, 0], b[:, 1]) <= m.cfg.range_m]
        med, p90, n = local_plan(b)
        print(f"{tag + ' / 单关键帧':>26} {n:>8} {med:>10.4f} {p90:>8.4f}   深度噪声下限")
        # 相邻两关键帧：位姿误差 ≈ 1 s 内的漂移。开头几帧附近没墙，要往后找一对真有墙的。
        pi = next((i for i in range(5, len(frames) - 1)
                   if len(clouds(frames[i:i + 2], m, which, sel)) > 500), None)
        if pi is None:
            print(f"{tag + ' / 相邻两帧':>26} {'—':>8}  找不到有墙的相邻帧对")
        else:
            pair = clouds(frames[pi:pi + 2], m, which, sel)
            med, p90, n = local_plan(pair)
            print(f"{tag + f' / 相邻两帧({pi},{pi+1})':>26} {n:>8} {med:>10.4f} {p90:>8.4f}")
        # 全云墙点
        t0 = time.perf_counter()
        allp = clouds(frames, m, which, sel)
        med, p90, n = local_plan(allp)
        verdict = "过止损线" if med <= STOP_LOSS_M else "不过止损线"
        print(f"{tag + ' / 全云墙点':>26} {n:>8} {med:>10.4f} {p90:>8.4f}   {verdict}"
              f"  ({time.perf_counter() - t0:.0f}s)")
        if which != 3:
            continue
        # 融合帧数扫描：既然 2 帧比 1 帧好、391 帧比 2 帧差，甜点必然在中间。
        # "融合越多越糊"就是位姿漂移随时间累积的直接证据，甜点位置则告诉我们
        # 一块墙最多能跨多少秒还保得住平面。
        print(f"\n  [{tag}] 沿轨迹均匀取样，扫融合帧数：")
        print(f"  {'帧数':>6} {'跨时长':>8} {'墙点数':>8} {'平面性中位':>10} {'p90':>8}")
        N = len(frames)
        span = np.linspace(0, N - 1, 50).round().astype(int)   # 50 个采样点估算跨时长
        for nf in (1, 2, 3, 5, 8, 12, 20, 30, 50, 100, 200, N):
            if nf > N:
                continue
            idx = np.linspace(0, N - 1, nf).round().astype(int)
            P = clouds([frames[i] for i in idx], m, which, sel)
            if len(P) < 50:
                continue
            med, p90, n = local_plan(P)
            dur = (span[-1] - span[0]) * 1.0 * (nf - 1) / max(len(span) - 1, 1)
            print(f"  {nf:>6} {dur:>7.0f}s {n:>8} {med:>10.4f} {p90:>8.4f}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    main()
