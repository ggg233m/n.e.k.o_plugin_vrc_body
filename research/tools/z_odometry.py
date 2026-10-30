#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""1-DOF 高度里程：只解相邻关键帧的竖直位移 DeltaZ，输出带 z 的位姿表。

为什么需要它
------------
位姿表 T[2,3] 恒 0（enable_vertical=False），所有楼层/挑空被压成一张平面。
OSC 的 vert.jsonl 只有坠落（66.7% 负速度、最长上升 0.39 s），够不到 2 m 层高。
但**深度图本身**含竖直信息：把两帧的点投到世界系，按水平格取高度中位数，
同格之差就是这两帧之间的竖直位移。

口径
----
* 用**世界系** z（p @ R.T），不是相机系 —— 相机 pitch 最大 53 度，
  用相机系 z 会把它整个吃进高度里（实测漂到 +168 m）。
* 只用**水平格**（格内 z 的 p90-p10 <= 0.05 m）。竖直面（墙）的可见范围
  随走动变化，混进来会漂（实测 +32.8 m）。
* 单步物理上限 max_step_m：关键帧 xy 间距 <=1.7 m，竖直不可能一步跳更多。

用法
----
    python research/tools/z_odometry.py --rec 20261029_060258 --report-only
    python research/tools/z_odometry.py --rec 20261029_060258 --write --anchor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_poses(rec_dir):
    """优先 poses_full.npz（含录制档末段），否则 poses.npz。"""
    for name in ("poses_full.npz", "poses.npz"):
        p = rec_dir / name
        if p.exists():
            with np.load(p) as z:
                return {int(k): np.asarray(T, np.float64) for k, T in zip(z["ids"], z["T"])}
    raise SystemExit("[fail] %s 没有任何位姿表" % rec_dir)


def frame_cells(rec_dir, k, T, range_m, cell_m, min_pts, max_spread):
    """一帧 -> {世界xy格: 该格的高度中位数}，只保留水平格。"""
    p = rec_dir / ("kf/%06d.npz" % k)
    if not p.exists():
        return {}
    with np.load(p) as z:
        pts = np.asarray(z["pts"], np.float32)
    ok = np.isfinite(pts).all(1) & (np.hypot(pts[:, 0], pts[:, 1]) <= float(range_m))
    pts = pts[ok]
    if len(pts) < min_pts:
        return {}
    pw = pts @ T[:3, :3].T
    xy = pw[:, :2] + T[:2, 3]
    zw = pw[:, 2]
    b = np.floor(xy / cell_m).astype(np.int64)
    key = b[:, 0].astype(np.int64) * 100000 + b[:, 1]
    order = np.argsort(key, kind="stable")
    key, zw = key[order], zw[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(key)) + 1, len(key)]
    out = {}
    for i in range(len(cuts) - 1):
        s = zw[cuts[i]:cuts[i + 1]]
        if len(s) < min_pts:
            continue
        if float(np.percentile(s, 90) - np.percentile(s, 10)) > max_spread:
            continue
        out[int(key[cuts[i]])] = float(np.median(s))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rec", required=True)
    ap.add_argument("--cell", type=float, default=0.15)
    ap.add_argument("--range-m", type=float, default=5.0)
    ap.add_argument("--min-pts", type=int, default=8)
    ap.add_argument("--max-spread", type=float, default=0.05)
    ap.add_argument("--max-step", type=float, default=0.8)
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--anchor-gap-m", type=float, default=150.0)
    ap.add_argument("--anchor-min-cells", type=int, default=200)
    ap.add_argument("--anchor-stride", type=int, default=25)
    ap.add_argument("--out", default="poses_z.npz")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--report-only", action="store_true")
    g.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rec_dir = ROOT / "navmesh_recordings" / args.rec
    if not rec_dir.is_dir():
        raise SystemExit("[fail] 没有这个录制：%s" % rec_dir)
    pose = load_poses(rec_dir)
    ks = sorted(pose)
    print("[load ] %s  %d 帧 (k=%d..%d)" % (args.rec, len(ks), ks[0], ks[-1]), flush=True)

    dm = {}
    ev = rec_dir / "events.jsonl"
    if ev.exists():
        for line in ev.open(encoding="utf-8"):
            e = json.loads(line)
            if e.get("kind") == "kf" and e.get("dist_m") is not None:
                dm[e["k"]] = float(e["dist_m"])

    cache_p = ROOT / ".tmp" / "z_odom" / (args.rec + "_cells.npz")
    cells = {}
    if cache_p.exists():
        with np.load(cache_p) as z:
            for key in z.files:
                k = int(key.split("_")[1])
                arr = z[key]
                cells[k] = {int(a): float(b) for a, b in arr}
        print("[cache] %s  %d 帧" % (cache_p.name, len(cells)), flush=True)
    needed = [k for k in ks if k not in cells]
    if needed:
        print("[cells] 计算 %d 帧的水平格 ..." % len(needed), flush=True)
        for i, k in enumerate(needed):
            cells[k] = frame_cells(rec_dir, k, pose[k], args.range_m,
                                   args.cell, args.min_pts, args.max_spread)
            if (i + 1) % 200 == 0:
                print("        %d/%d" % (i + 1, len(needed)), flush=True)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_p, **{("k_%d" % k): np.array(sorted(cells[k].items()))
                                        for k in sorted(cells) if cells[k]})
        print("[cache] 写入 %s" % cache_p.name, flush=True)

    def dz(k0, k1, trim=0.25):
        a, b = cells.get(k0, {}), cells.get(k1, {})
        common = [kk for kk in b if kk in a]
        if not common:
            return float("nan"), 0
        d = np.sort(np.array([a[kk] - b[kk] for kk in common]))
        n = len(d)
        lo = int(n * trim)
        return (float(np.mean(d[lo:n - lo])) if n > 2 * lo else float(np.median(d))), n

    print("[odom ] 相邻帧 DeltaZ ...", flush=True)
    steps = {}
    nobs = 0
    for k0, k1 in zip(ks, ks[1:]):
        d, n = dz(k0, k1)
        if not np.isfinite(d) or n < 20:
            nobs += 1
            continue
        steps[k1] = float(np.clip(d, -args.max_step, args.max_step))
    print("        有效 %d / %d   无观测 %d" % (len(steps), len(ks) - 1, nobs), flush=True)

    still = [k for k in steps if k in dm and (k - 1) in dm and abs(dm[k] - dm[k - 1]) < 0.02]
    if still:
        v = np.array([steps[k] for k in still])
        print("[check] 低运动段(真值应0): n=%d 中位 %+.4f 平均 %+.4f |dz|中位 %.4f"
              % (len(v), np.median(v), v.mean(), np.median(np.abs(v))), flush=True)

    idx = {k: i for i, k in enumerate(ks)}
    n = len(ks)
    rows, rhs, wts = [], [], []
    for k1, d in steps.items():
        k0 = ks[idx[k1] - 1]
        r = np.zeros(n)
        r[idx[k0]] = -1.0
        r[idx[k1]] = 1.0
        rows.append(r)
        rhs.append(d)
        wts.append(1.0)

    anchors = []
    if args.anchor:
        sub = [k for k in ks if k % args.anchor_stride == 0]
        print("[anchor] 扫描 %d 帧 ..." % len(sub), flush=True)
        for ai, i in enumerate(sub):
            for j in sub[:ai]:
                if abs(dm.get(i, 0.0) - dm.get(j, 0.0)) < args.anchor_gap_m:
                    continue
                d, cnt = dz(j, i)
                if np.isfinite(d) and cnt >= args.anchor_min_cells:
                    anchors.append((j, i, d, cnt))
        best = {}
        for a in anchors:
            if a[1] not in best or a[3] > best[a[1]][3]:
                best[a[1]] = a
        anchors = sorted(best.values())
        print("         保留 %d 条锚点" % len(anchors), flush=True)
        for a, b, d, c in anchors:
            r = np.zeros(n)
            r[idx[a]] = -1.0
            r[idx[b]] = 1.0
            rows.append(r)
            rhs.append(d)
            wts.append(5.0)

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr
    A = np.asarray(rows)
    b = np.asarray(rhs)
    w = np.asarray(wts)
    sol = lsqr(csr_matrix(A * w[:, None]), b * w, damp=1e-3, iter_lim=20000)
    z = sol[0] - sol[0][0]

    print("[solve] z 范围 %+.2f .. %+.2f m  std %.2f" % (z.min(), z.max(), z.std()), flush=True)
    if anchors:
        err = np.array([abs((z[idx[bb]] - z[idx[a]]) - d) for a, bb, d, c in anchors])
        print("[solve] 锚点残差: 中位 %.3f  max %.3f m" % (np.median(err), err.max()), flush=True)

    for i in range(0, len(ks), max(1, len(ks) // 16)):
        print("        k=%5d  z=%+7.2f" % (ks[i], z[i]), flush=True)

    if args.write:
        T = np.stack([pose[k] for k in ks]).astype(np.float64)
        for i, k in enumerate(ks):
            T[i][2, 3] = z[i]
        out = rec_dir / args.out
        np.savez_compressed(out, ids=np.asarray(ks, np.int64), T=T)
        print("[out  ] %s  %d 帧" % (out, len(ks)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
