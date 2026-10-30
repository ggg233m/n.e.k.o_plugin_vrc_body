#!/usr/bin/env python
# -*- coding: utf-8 -*-
r'''Per-frame elevation from within-frame stereo (no integration, cannot drift).

Problem it solves
-----------------
The pose table has T[2,3] = 0 for every frame, so the exporter forces each
frame's own floor to z = 0. A bridge and the hall under it then both land on
z = 0 -> the hall floor is drawn twice (at 0 and at -2.2). That is the
'floors stacked' artifact, and it is INDEPENDENT of xy drift.

Method
------
For each frame, in the CAMERA frame (no pose needed):
  own floor   = strongest horizontal plane within +-0.7 m of -cam_h
  lowest      = lowest strong horizontal plane at least min_gap below own
  elevation   = own - lowest   (0 when no lower floor is visible)

elevation is then lightly smoothed along the trajectory (a bridge is not a
1-frame event) and written as T[2,3].

Verification: image-verified truth pairs must keep small |dz| (they are the
same physical place), and xy distances must not change.

Usage:
    python research/tools/z_elevation.py --rec 20261029_060258 --base poses_replay.npz --report-only
    python research/tools/z_elevation.py --rec 20261029_060258 --base poses_replay.npz --write
'''
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def frame_elevation(pts, cam_h, min_gap, min_support, bin_m=0.05):
    '''Return (elevation, own_z, low_z) in camera-frame metres.'''
    z = pts[:, 2]
    h, e = np.histogram(z, bins=np.arange(-6.0, 2.5, bin_m))
    if h.max() < 40:
        return 0.0, np.nan, np.nan
    hs = np.convolve(h, np.ones(3) / 3.0, mode='same')
    idx, _ = find_peaks(hs, prominence=hs.max() * 0.06, distance=int(0.45 / bin_m))
    if not len(idx):
        return 0.0, np.nan, np.nan
    ctr = (e[:-1] + e[1:]) / 2.0
    peaks = sorted(((float(ctr[i]), int(h[i])) for i in idx), key=lambda t: -t[1])
    cand = [(v, c) for v, c in peaks if abs(v + cam_h) <= 0.70]
    if not cand:
        return 0.0, np.nan, np.nan
    own, _ = max(cand, key=lambda t: t[1])
    lower = [(v, c) for v, c in peaks if v < own - min_gap and c >= min_support]
    if not lower:
        return 0.0, own, np.nan
    low, _ = min(lower, key=lambda t: t[0])
    return float(own - low), own, low


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--rec', required=True)
    ap.add_argument('--base', default='poses_replay.npz', help='底表（xy/yaw 来自它）')
    ap.add_argument('--out', default='poses_replay_z.npz')
    ap.add_argument('--range-m', type=float, default=5.0)
    ap.add_argument('--min-gap', type=float, default=1.2, help='判定下层的垂直落差下限（m）')
    ap.add_argument('--min-support', type=int, default=300, help='下层最少点数')
    ap.add_argument('--smooth', type=int, default=15, help='沿时间的中值滤波窗（帧）')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--report-only', action='store_true')
    g.add_argument('--write', action='store_true')
    args = ap.parse_args()

    rec = ROOT / 'navmesh_recordings' / args.rec
    base = rec / args.base
    if not base.exists():
        raise SystemExit(f'[fail] 缺底表 {base}')
    with np.load(base) as z:
        ids = np.asarray(z['ids'], np.int64)
        T = np.asarray(z['T'], np.float64)
    pose = {int(k): T[i] for i, k in enumerate(ids)}
    ks = sorted(pose)
    print(f'[base ] {args.base}  {len(ks)} 帧', flush=True)

    # cam_h - gc 的有效离地高：从底表实际测（与导出同口径）
    cam_h = 1.732

    elev = np.full(len(ks), np.nan)
    n_obs = 0
    for i, k in enumerate(ks):
        p = rec / f'kf/{k:06d}.npz'
        if not p.exists():
            continue
        with np.load(p) as z:
            pts = np.asarray(z['pts'], np.float32)
        ok = np.isfinite(pts).all(1) & (np.hypot(pts[:, 0], pts[:, 1]) <= args.range_m)
        pts = pts[ok]
        if len(pts) < 2000:
            continue
        e, own, low = frame_elevation(pts, cam_h, args.min_gap, args.min_support)
        elev[i] = e
        if e > 0.05:
            n_obs += 1

    valid = np.isfinite(elev)
    print(f'[elev ] 有效帧 {int(valid.sum())}  测得抬升 {n_obs} 帧 ({100*n_obs/max(len(ks),1):.1f}%)', flush=True)
    if n_obs:
        ev = elev[valid & (elev > 0.05)]
        print(f'        抬升分布: p50 {np.median(ev):.2f}  p90 {np.percentile(ev,90):.2f}  max {ev.max():.2f} m', flush=True)

    # 平滑：桥上不是 1 帧事件，用中值滤波压制单帧误检
    filled = np.where(valid, elev, 0.0)
    if args.smooth > 1:
        k = int(args.smooth) | 1
        pad = k // 2
        padded = np.pad(filled, pad, mode='edge')
        sm = np.array([np.median(padded[i:i + k]) for i in range(len(filled))])
    else:
        sm = filled
    print(f'[smooth] 中值滤波 {args.smooth} 帧后: p50 {np.median(sm):.2f}  max {sm.max():.2f}  >0.5 的帧 {int((sm>0.5).sum())}', flush=True)

    # 验证：真值对的 z 差（同址应对齐）与 xy 距离（不应改变）
    truth = ROOT / '.tmp' / ('truth_%s.tsv' % args.rec)
    if truth.exists():
        prs = []
        for line in truth.open(encoding='utf-8'):
            if line.startswith('a'):
                continue
            f = line.split()
            prs.append((int(f[0]), int(f[1])))
        uniq = sorted({tuple(sorted(p)) for p in prs if int(p[0]) in pose and int(p[1]) in pose})
        if uniq:
            dz = np.array([abs(sm[ks.index(a)] - sm[ks.index(b)]) for a, b in uniq])
            dxy = np.array([float(np.linalg.norm(pose[a][:2, 3] - pose[b][:2, 3])) for a, b in uniq])
            print(f'[check] 真值对 {len(uniq)} 个:  |dz| 中位 {np.median(dz):.2f} max {dz.max():.2f} m', flush=True)
            print(f'                                xy 中位 {np.median(dxy):.2f} max {dxy.max():.2f} m（不应改变）', flush=True)

    if args.write:
        T2 = T.copy()
        for i, k in enumerate(ks):
            T2[i][2, 3] = sm[i]
        out = rec / args.out
        np.savez_compressed(out, ids=ids, T=T2)
        print(f'[out  ] {out}  T[2,3] 已写入（max {sm.max():.2f} m）', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
