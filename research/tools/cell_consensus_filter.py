# -*- coding: utf-8 -*-
r"""逐格共识剔除：按同一 xy 格多数帧的 z 否决少数帧的离群点。

## 为什么需要它（2026-10-28 实测）

竖向叠层（同一 xy 格内 z 的 p5-p95 极差）实测：

| 口径 | 格数 | 极差 p50 | p75 | p90 |
|---|---|---|---|---|
| 原样 | 12922 | 18.23 cm | 35.37 | 49.93 |
| 共识剔除 0.15 m | 12003 | 10.82 | 15.26 | 20.30 |
| 共识剔除 0.10 m | 11431 | 9.09 | 11.89 | 14.34 |
| 共识剔除 0.05 m | 10009 | 5.49 | 7.02 | 8.21 |

阈值 0.10 m 时 p50 降 50%、p90 降 71%，代价是保留 86.2% 的点。

## 为什么逐格而不是逐帧

* 整帧地面估计（gc）的误差：15% 帧 >5 cm、8% >10 cm（最差 -0.43 m）
* 逐帧逐格的局部偏差：p50 2.4 cm、p90 18.3 cm、17.8% 的 (帧,格) 对 >10 cm

主导的不是「整帧偏了」，而是「少数帧在少数区域偏了」。逐帧一个标量原理上
表达不了这种偏差 —— 这是 global / raw / 身高门 / 整帧共识四种方案都失败的原因。

## 它与修正的区别

只删，不动。保留点位置不变，不会引入新的几何错误，代价只是覆盖率。

## 已知盲区

共识来自同一批数据。若某区域多数帧都朝同一方向偏（共模），多数票会选错。
审计实测地面众数落在 z=-0.025、[-0.15,0) 与 [0,+0.15) 各占约 20%，
这种「被劈成两半」的情形正是盲区。用之前先看 --report-only 的保留率与极差。

用法：
    python research/tools/cell_consensus_filter.py 20261026_184318 --report-only
    python research/tools/cell_consensus_filter.py 20261026_184318 --dump-mask .tmp/mask.npz
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CAM_H = 1.742
GC = -0.077
CELL = 0.25
OFF = 2000
STRIDE = 8
XYO, XYS, ZS, ZO = 32.0, 1000.0, 50.0, 1.0


def cell_key(x, y):
    gx = np.floor(np.asarray(x) / CELL).astype(np.int64) + OFF
    gy = np.floor(np.asarray(y) / CELL).astype(np.int64) + OFF
    return gx * 8192 + gy


def load_frames(rec, pose_npz, cam_h, gc):
    with np.load(rec / (pose_npz + '.npz')) as z:
        pose = {int(i): np.asarray(T, float) for i, T in zip(z['ids'], z['T'])}
    ev = [json.loads(s) for s in (rec / 'events.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    kfs = sorted([e for e in ev if e.get('kind') == 'kf' and not e.get('unmapped')], key=lambda e: e['k'])
    out = {}
    for e in kfs:
        k = int(e['k'])
        if k not in pose:
            continue
        with np.load(rec / 'kf' / ('%06d.npz' % k)) as z:
            P = np.asarray(z['pts'], np.float32)
        ok = np.isfinite(P).all(axis=1) & (np.hypot(P[:, 0], P[:, 1]) <= 5.0)
        Q = P[ok][::4]
        if len(Q) < 200:
            continue
        T = pose[k]
        pw = Q.astype(np.float64) @ T[:3, :3].T
        xy = pw[:, :2] + T[:2, 3]
        zw = pw[:, 2] + float(T[2, 3]) + cam_h - gc
        band = (zw > -0.6) & (zw < 0.6)
        if int(band.sum()) < 200:
            continue
        X, Y, Z = xy[band, 0], xy[band, 1], zw[band]
        out[k] = (X, Y, Z, cell_key(X, Y))
    return out


def build_consensus(frames):
    bins = {}
    for _k, (X, Y, Z, key) in frames.items():
        u, inv = np.unique(key, return_inverse=True)
        for i in range(len(u)):
            m = inv == i
            if int(m.sum()) >= 6:
                bins.setdefault(int(u[i]), []).append(float(np.median(Z[m])))
    return {kk: float(np.median(v)) for kk, v in bins.items() if len(v) >= 6}


def report(frames, cons, thrs):
    print('  帧 %d  共识格 %d' % (len(frames), len(cons)))
    print()
    print('  %6s %10s %10s %8s %14s' % ('thr', 'kept', 'total', 'kept%', 'cell p50'))
    for thr in thrs:
        bins = {}
        kept = total = 0
        for _k, (X, Y, Z, key) in frames.items():
            total += len(Z)
            keep = np.ones(len(Z), bool)
            for kk in np.unique(key):
                c = cons.get(int(kk))
                if c is None:
                    continue
                keep &= ~((key == kk) & (np.abs(Z - c) > thr))
            kept += int(keep.sum())
            kk2, inv2 = np.unique(key[keep], return_inverse=True)
            for i in range(len(kk2)):
                m = inv2 == i
                if int(m.sum()) >= 4:
                    bins.setdefault(int(kk2[i]), []).append(float(np.median(Z[keep][m])))
        rng = []
        for kk, v in bins.items():
            if len(v) >= 6:
                a = np.asarray(v)
                rng.append(float(np.percentile(a, 95) - np.percentile(a, 5)))
        med = np.median(rng) * 100 if rng else float('nan')
        print('  %6.2f %10d %10d %7.1f%% %13.2fcm' % (thr, kept, total, 100.0 * kept / max(total, 1), med))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('rec')
    ap.add_argument('--pose', default='final_poses')
    ap.add_argument('--cam-h', type=float, default=CAM_H)
    ap.add_argument('--gc', type=float, default=GC)
    ap.add_argument('--thr', type=float, default=0.10)
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--dump-mask', type=Path)
    ap.add_argument('--apply-html', type=Path, help='按共识剔除该 HTML 的点并重写')
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()

    rec = ROOT / 'navmesh_recordings' / a.rec
    if not rec.is_dir():
        raise SystemExit('[fail] no such recording: %s' % rec)
    print('[%s] pose=%s cam_h=%.3f gc=%.3f' % (a.rec, a.pose, a.cam_h, a.gc))
    frames = load_frames(rec, a.pose, a.cam_h, a.gc)
    cons = build_consensus(frames)
    if not cons:
        raise SystemExit('[fail] 没有足够的共识格（每格需 >=6 帧）')
    report(frames, cons, (0.05, 0.10, 0.15, 0.20))

    if a.dump_mask:
        masks = {}
        for k, (X, Y, Z, key) in frames.items():
            keep = np.ones(len(Z), bool)
            for kk in np.unique(key):
                c = cons.get(int(kk))
                if c is None:
                    continue
                keep &= ~((key == kk) & (np.abs(Z - c) > a.thr))
            masks[str(k)] = keep
        np.savez_compressed(a.dump_mask, **masks)
        print('  mask -> %s' % a.dump_mask)

    if a.apply_html is not None:
        if a.out is None:
            raise SystemExit('[fail] --apply-html 需要 --out')
        h = a.apply_html.read_text(encoding='utf-8', errors='replace')
        m = re.search(r'const RAW = atob\("([A-Za-z0-9+/=]*)"\)', h)
        if not m:
            raise SystemExit('[fail] 没有 RAW 载荷')
        raw = np.frombuffer(base64.b64decode(m.group(1)), np.uint8).reshape(-1, STRIDE)
        x = (raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8)) / XYS - XYO
        y = (raw[:, 2].astype(np.int32) | (raw[:, 3].astype(np.int32) << 8)) / XYS - XYO
        z = raw[:, 4].astype(np.float64) / ZS - ZO
        key = cell_key(x, y)
        keep = np.ones(len(raw), bool)
        for kk in np.unique(key):
            c = cons.get(int(kk))
            if c is None:
                continue
            keep &= ~((key == kk) & (np.abs(z - c) > a.thr))
        print()
        print('  HTML 点 %d -> %d (%.1f%%)' % (len(raw), int(keep.sum()), 100.0 * keep.mean()))
        print('  NOTE: HTML 点是已体素化的 0.09 m，与 build_consensus 的 0.25 m 格不同源；')
        print('        这里按 HTML 点自己的 xy 落格查共识，只对「该格有共识」的点生效。')
        return 0

    print()
    print('  WARN: 共识来自同一批数据。若某区域多数帧都朝同一方向偏（共模），多数票会选错。')
    print('        先看上面的保留率与格内极差 —— 极差没降说明该区没有一致意见。')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())