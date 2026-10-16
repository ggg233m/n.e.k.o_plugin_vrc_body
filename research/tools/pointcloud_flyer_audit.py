# -*- coding: utf-8 -*-
"""点云飞点审计：把「孤立度」和它的**所有反面证据**一次性算出来。

为什么必须合并成一个工具
------------------------
孤立度单独看会误导：真实的细窄物体（棕榈叶、吊灯串、柱子棱边）本身就稀疏，
看起来和伪影一模一样。任何"孤立就滤掉"的方案都会误伤它们。所以本工具
**刻意把误伤率与收益并排输出**，且先跑一个注入验证 —— 判据不响应就直接报错退出。

用法::

    python research/tools/pointcloud_flyer_audit.py --rec 20261013_223033 \\
        --session 20261013_223033 --world wrld_home-7cf435ea

输出（全部在同一份点云上，口径与 pointcloud_view 一致）：
  1. 注入验证          判据对已知误差是否响应（不通过则警告）
  2. 孤立度总体分布
  3. 高度分档          地板带 vs 其余（最强的判别信号）
  4. hits 分档         体素被多少帧看到
  5. 收益/代价表       hits 过滤对「孤立点」和「地板点」的分别影响
  6. 身体体积自标注    轨迹管内的点（已证明为空），及其与孤立判据的 IoU

已知边界（见 Docs/空中飞点归因与清理判据实测（2026-10-16）.md）
--------------------------------------------------------------
* 体素合并本身会制造稀疏：0.0675 m 体素下被合并成 1 点的格子看起来就是孤立。
* 孤立 ≠ 伪影。本工具只给数字，**不做删除决定**。
* 身体体积判据覆盖不到没走过的地方（实测 IoU 仅 2.3%）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "tools"))

import pointcloud_view as V  # noqa: E402

FLOOR_BAND_M = 0.30      # |z| 小于此值算地板带（与 prior_topview 同口径）
ISO_K = 9                # 取第 (K-1) 近邻
ISO_THR_M = 0.15         # 孤立阈值


def build(world_dir: Path, rec: str, sid: str, vox: float, max_pts: int = 1_500_000):
    """与 pointcloud_view.load_session 同口径重建点云，并额外带出**帧号**。

    为什么不直接读 viewer 的 HTML：那里没有帧号，而 hits / 逐帧归因都要它。
    """
    cam_h = V.prior_cam_h(world_dir).get(sid) or V.estimate_cam_h(ROOT / "navmesh_recordings" / rec)
    pose_of, pose_src = V.pose_table_rigid(world_dir, sid)
    kfs = sorted((ROOT / "navmesh_recordings" / rec).glob("kf/*.npz"), key=lambda p: int(p.stem))
    ch, ki, ci = [], [], []
    n_raw = 0
    for kf in kfs:
        k = int(kf.stem)
        T = pose_of.get(k)
        if T is None:
            continue
        with np.load(kf) as z:
            p = np.asarray(z["pts"], np.float32)
        n_raw += len(p)
        ok = np.isfinite(p).all(1) & (np.hypot(p[:, 0], p[:, 1]) <= 5.0)
        p = p[ok]
        if not len(p):
            continue
        gc = V.frame_ground_offset(p, T, cam_h)
        pw = p @ T[:3, :3].T
        xy = pw[:, :2] + T[:2, 3]
        zw = pw[:, 2] + cam_h - gc
        sel = (zw > -1.0) & (zw < 4.0) & np.isfinite(xy).all(1)
        if not sel.any():
            continue
        n = int(sel.sum())
        ch.append(np.column_stack([xy[sel], zw[sel]]))
        ki.append(np.full(n, k, np.int32))
        ci.append(np.tile(np.asarray(T)[:3, 3], (n, 1)))    # 每点的**相机位置**（世界系）
    P = np.concatenate(ch).astype(np.float64)
    K = np.concatenate(ki)
    CAMO = np.concatenate(ci).astype(np.float64)
    # 体素降采样（质心），帧号取格内首个出现的 —— 与 sid 同规则，保证可复现
    q = np.floor(P / vox).astype(np.int64)
    key = q[:, 0] * (1 << 42) + q[:, 1] * (1 << 21) + q[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    n = len(uk)
    acc = np.zeros((n, 3))
    np.add.at(acc, inv, P)
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    Pv = acc / cnt[:, None]
    Kv = np.zeros(n, np.int32)
    Kv[inv[::-1]] = K[::-1]
    # hits = 该体素被多少个**不同帧**看到（不是多少点）
    pair = inv.astype(np.int64) * (1 << 20) + K
    hv = np.bincount(np.unique(pair) // (1 << 20), minlength=n).astype(np.int32)
    CV = np.zeros((n, 3))
    CV[inv[::-1]] = CAMO[::-1]        # 与 Kv 同规则：格内取首次出现的
    return Pv, Kv, hv, CV, cam_h, pose_src, n_raw, cnt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--session", default=None)
    ap.add_argument("--vox", type=float, default=0.0675)
    a = ap.parse_args()

    wdir = Path(a.world)
    if not wdir.is_absolute():
        wdir = ROOT / "navmesh_memory" / a.world
    sid = a.session or a.rec
    t0 = time.perf_counter()
    P, K, hits, CAMO, cam_h, pose_src, n_raw, voxcnt = build(wdir, a.rec, sid, a.vox)
    print("[build] %s  原始 %d 点 → 体素 %d 点  帧 %d  (%.1fs)"
          % (a.rec, n_raw, len(P), len(set(K.tolist())), time.perf_counter() - t0))
    print("[build] cam_h=%.3f  pose=%s" % (cam_h, pose_src))

    # ---------- 1. 注入验证 ----------
    print()
    print("=== 1. 注入验证（判据不响应就没资格用来下结论）===")
    rng = np.random.default_rng(0)
    nb = rng.choice(len(P), min(20000, len(P)), replace=False)
    t = cKDTree(P)
    d0, _ = t.query(P[nb], k=ISO_K, workers=-1)
    base = float(np.median(d0[:, -1]))
    # ⚠️ 必须沿**该点自己的视线**外推（相机 → 点），不能用点云质心：
    # 质心方向几乎随机，实测只有 1.12 倍，会把有效判据误判为无效。
    u = P[nb] - CAMO[nb]
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    print("  原样   第 %d 近邻距离 p50 = %.4f m" % (ISO_K - 1, base))
    print("  沿视线外推的响应（扫描 —— 单一幅度会掩盖灵敏度边界）：")
    for amp in (0.30, 0.60, 1.00, 1.50):
        P2 = P.copy()
        P2[nb] = P2[nb] + u * amp
        t2 = cKDTree(P2)
        d1, _ = t2.query(P2[nb], k=ISO_K, workers=-1)
        inj = float(np.median(d1[:, -1]))
        print("     +%.2f m → %.4f m  （%.2f 倍）%s"
              % (amp, inj, inj / max(base, 1e-9), "   ← 超过阈值，能被标出" if inj > ISO_THR_M else ""))
        if amp == 0.30:
            near_thr = inj > ISO_THR_M
        del P2, t2
    print("  ⚠️ **判据的灵敏度边界**：外推量必须让「第 %d 邻距离」超过 %.2f m 才会被标出。"
          % (ISO_K - 1, ISO_THR_M))
    print("     0.30 m 量级的位移**不会**被本判据捕获（实测 %.2f 倍）⇒")
    print("     本判据只对「点被丢进无邻居的空旷处」敏感，对「点在稠密区里小幅度偏移」不敏感。")
    print("     ⇒ 用它抓混合像素是**间接**的（混合像素往往飞得远），不是直接检测。")

    # ---------- 2. 孤立度 ----------
    d, _ = t.query(P, k=ISO_K, workers=-1)
    r8 = d[:, -1]
    iso = r8 > ISO_THR_M
    print()
    print("=== 2. 孤立度（第 %d 近邻 > %.2f m）===" % (ISO_K - 1, ISO_THR_M))
    print("  孤立 %d / %d = %.3f%%" % (int(iso.sum()), len(P), 100.0 * iso.mean()))
    print("  r8: p50=%.4f p90=%.4f p99=%.4f max=%.4f"
          % (np.median(r8), np.percentile(r8, 90), np.percentile(r8, 99), r8.max()))
    for th in (0.10, 0.15, 0.20, 0.30, 0.50):
        print("     r8 > %.2f m : %8d  %7.4f%%"
              % (th, int((r8 > th).sum()), 100.0 * (r8 > th).mean()))

    # ---------- 3. 高度分档 ----------
    print()
    print("=== 3. 高度分档（地板带 vs 其余 —— 最强的判别信号）===")
    print("  %-16s %10s %10s" % ("世界高度", "占全部点", "孤立率"))
    for lo, hi in ((-1.0, -0.3), (-0.3, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.1)):
        m = (P[:, 2] >= lo) & (P[:, 2] < hi)
        if m.sum() < 100:
            continue
        tag = "  ← 地板带" if (lo <= 0 <= hi) else ""
        print("  %+5.1f..%+5.1f m %9.1f%% %9.3f%%%s"
              % (lo, hi, 100.0 * m.mean(), 100.0 * iso[m].mean(), tag))

    # ---------- 4. hits ----------
    print()
    print("=== 4. hits（体素被多少个不同帧看到）===")
    u2, c2 = np.unique(hits, return_counts=True)
    for x, y in list(zip(u2[:6], c2[:6])):
        print("     hits=%-4d %8d  (%.2f%%)" % (x, y, 100.0 * y / len(P)))
    print("  中位 %d  p90 %d  max %d" % (np.median(hits), np.percentile(hits, 90), hits.max()))

    # ---------- 5. 收益/代价 ----------
    print()
    print("=== 5. hits 过滤的收益与代价（⚠️ 代价栏是红的就别用）===")
    flr = np.abs(P[:, 2]) < FLOOR_BAND_M
    print("  %-6s %10s %12s %14s %14s"
          % ("阈值", "保留点%", "孤立消除%", "地板点滤掉%", "地板格损失%"))
    q0 = np.floor(P[flr][:, :2] / 0.15).astype(np.int64)
    base_cells = len(np.unique(q0[:, 0] * (1 << 21) + q0[:, 1]))
    for th in (1, 2, 3, 4, 6):
        keep = hits >= th
        fl = flr & keep
        q1 = np.floor(P[fl][:, :2] / 0.15).astype(np.int64)
        cells = len(np.unique(q1[:, 0] * (1 << 21) + q1[:, 1])) if fl.sum() else 0
        print("  %-6d %9.2f%% %11.2f%% %13.2f%% %13.2f%%"
              % (th, 100.0 * keep.mean(),
                 100.0 * (1 - iso[keep].sum() / max(int(iso.sum()), 1)),
                 100.0 * (flr & ~keep).sum() / max(int(flr.sum()), 1),
                 100.0 * (cells - base_cells) / max(base_cells, 1)))

    # ---------- 6. 身体体积自标注 ----------
    print()
    print("=== 6. 身体体积自标注（轨迹管内 = 已证明为空，不需要阈值）===")
    # ⚠️ 必须用**相机位置**（T 的平移），不是每帧点云的质心。
    # 质心会被该帧看到的东西带偏（尤其近场大量地板点），管的位置就错了：
    # 实测用质心时体内点数虚高到 131,908，正解是 39,568。
    ks = sorted(set(K.tolist()))
    CAM = np.array([CAMO[K == k][0] for k in ks])
    seg = np.linalg.norm(np.diff(CAM[:, :2], axis=0), axis=1)
    dense = []
    for i in range(len(CAM) - 1):
        nseg = max(1, int(np.ceil(seg[i] / 0.05)))
        tt = np.linspace(0, 1, nseg, endpoint=False)
        dense.append(CAM[i, :2] + (CAM[i + 1, :2] - CAM[i, :2]) * tt[:, None])
    dense.append(CAM[-1:, :2])
    TD = np.vstack(dense)
    print("  轨迹折线 %.1f m → 重采样 %d 点" % (seg.sum(), len(TD)))
    tt2 = cKDTree(TD)
    for rad in (0.25,):
        near = tt2.query_ball_point(P[:, :2], rad)
        horiz = np.zeros(len(P), bool)
        for i, idx in enumerate(near):
            if idx:
                horiz[i] = True
        body = horiz & (P[:, 2] >= 0.20) & (P[:, 2] <= 1.70)
        inter = int((body & iso).sum())
        union = int((body | iso).sum())
        print("  管半径 %.2f m，高度 0.20–1.70 m：体内 %d 点 (%.3f%%)，其中孤立 %d"
              % (rad, int(body.sum()), 100.0 * body.mean(), inter))
        print("  与孤立判据 IoU = %.1f%%  ⇒ %s"
              % (100.0 * inter / max(union, 1),
                 "覆盖有限，只能当局部铁证" if inter / max(union, 1) < 0.3 else "覆盖较大"))
        for lo, hi in ((-1.0, -0.15), (-0.15, 0.15), (0.15, 0.40), (0.40, 0.80),
                       (0.80, 1.20), (1.20, 1.75), (1.75, 4.10)):
            m = horiz & (P[:, 2] >= lo) & (P[:, 2] < hi)
            if m.sum() < 50:
                continue
            tag = "  ← 地板（真实，非伪影）" if lo <= 0 <= hi else ""
            print("     管内 %-14s %8d  其中孤立 %5d%s"
                  % ("%+.2f..%+.2f m" % (lo, hi), int(m.sum()), int((m & iso).sum()), tag))
    print()
    print("⚠️ 本工具只给数字，不做删除决定。孤立 ≠ 伪影（真实细窄物体也孤立）。")


if __name__ == "__main__":
    main()
