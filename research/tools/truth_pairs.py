#!/usr/bin/env python
# -*- coding: utf-8 -*-
r'''Truth-pair builder: image-verified same-place keyframe pairs.

Why this tool exists
-------------------
Every geometry metric the repo has tried so far either
  * measures the player's height (own-floor z) - confounded by platforms, or
  * measures surface thickness - insensitive to a 14 m registration error, or
  * needs a pose table to build candidate pairs - circular when the pose is wrong.

This tool builds pairs from IMAGE CONTENT ONLY (ORB + mutual matching), with
two independent geometry checks that use no pose at all:
  1. PnP with the map frame's OWN 3-D points (des3d) - needs >=min_inliers;
  2. coverage of the image by the inliers.

Output: pairs TSV, usable as an external yardstick for any pose table.

Usage:
    python research/tools/truth_pairs.py --rec 20261029_060258 --out .tmp/truth.tsv
'''
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nav_loop import LoopConfig, relative_pose  # noqa: E402
from research.tools.loop_replay import features_of       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--rec', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--min-path-m', type=float, default=150.0,
                    help='两帧路程差下限：太近的不会漂，没有判别力')
    ap.add_argument('--min-inliers', type=int, default=200,
                    help='比回环门(50)严得多，只留高质量对')
    ap.add_argument('--min-coverage', type=float, default=0.30)
    ap.add_argument('--max-cands', type=int, default=8,
                    help='bow 候选数')
    args = ap.parse_args()

    rec = ROOT / 'navmesh_recordings' / args.rec
    kfs = sorted(rec.glob('kf/*.npz'), key=lambda p: int(p.stem))
    if not kfs:
        raise SystemExit(f'[fail] {rec}/kf 是空的')
    print(f'[load ] {args.rec}  {len(kfs)} 帧', flush=True)

    cfg = LoopConfig()
    feats = {}
    dist = {}
    for f in kfs:
        k = int(f.stem)
        with np.load(f) as z:
            feats[k] = features_of(z)
            dist[k] = float(np.asarray(z['dist_m']).reshape(-1)[0])
    ks = sorted(feats)
    print('[feat ] 载入完成', flush=True)

    # 外观检索：用每一帧的 des 去全库找最像的，不需要位姿
    from backend.nav_bow import BowIndex, BowVocabulary
    bow = None
    vocab = ROOT / cfg.bow_vocab if cfg.bow_vocab else None
    if vocab and vocab.exists():
        try:
            bow = BowIndex(BowVocabulary.load(str(vocab)))
            for k in ks:
                bow.add(k, feats[k].des)
            print(f'[bow  ] 建库完成 {len(ks)} 帧', flush=True)
        except Exception as e:                                # pragma: no cover
            print(f'[bow  ] 不可用：{e}（退回按路程配对）', flush=True)
            bow = None

    pairs = []
    t0 = time.time()
    n_try = 0
    reasons = {}
    for i, b in enumerate(ks):
        if i % 50 == 0 and i:
            print(f'        {i}/{len(ks)}  已接受 {len(pairs)}  {time.time()-t0:.0f}s', flush=True)
        cands = []
        if bow is not None:
            try:
                for a, _score in bow.query(feats[b].des, top_n=args.max_cands * 4):
                    if a != b and abs(dist[b] - dist[a]) >= args.min_path_m:
                        cands.append(a)
                    if len(cands) >= args.max_cands:
                        break
            except Exception:
                cands = []
        if not cands:
            # 退路：按路程差找（与位姿无关，只依赖 OSC 路程，粗略但非循环）
            target = dist[b] - args.min_path_m
            j = int(np.searchsorted([dist[k] for k in ks], target))
            if 0 <= j < len(ks):
                cands = [ks[j]]
            target2 = dist[b] + args.min_path_m
            j2 = int(np.searchsorted([dist[k] for k in ks], target2))
            if 0 <= j2 < len(ks):
                cands.append(ks[j2])
        for a in cands:
            n_try += 1
            rel, why = relative_pose(feats[a], feats[b], cfg)
            if rel is None:
                reasons[why] = reasons.get(why, 0) + 1
                continue
            if rel['inliers'] < args.min_inliers:
                reasons['low_inliers'] = reasons.get('low_inliers', 0) + 1
                continue
            if rel['coverage'] < args.min_coverage:
                reasons['low_coverage'] = reasons.get('low_coverage', 0) + 1
                continue
            pairs.append((a, b, rel['inliers'], rel['coverage'],
                          round(rel['reproj_px'], 2), round(dist[b]-dist[a], 1)))

    print()
    print(f'[pair ] 试了 {n_try} 对，接受 {len(pairs)} 对', flush=True)
    print(f'        拒绝原因: {reasons}', flush=True)
    if pairs:
        inp = np.array([p[2] for p in pairs])
        cov = np.array([p[3] for p in pairs])
        gap = np.array([p[5] for p in pairs])
        print(f'        内点: 中位 {np.median(inp):.0f}  min {inp.min():.0f}', flush=True)
        print(f'        覆盖: 中位 {np.median(cov):.2f}  min {cov.min():.2f}', flush=True)
        print(f'        路程差: 中位 {np.median(gap):.0f} m  max {gap.max():.0f}', flush=True)
    if args.out:
        out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8') as f:
            f.write('a\tb\tinliers\tcoverage\treproj_px\tpath_gap_m\n')
            for p in sorted(pairs):
                f.write('%d\t%d\t%d\t%.3f\t%.2f\t%.1f\n' % p)
        print(f'[out  ] {out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
