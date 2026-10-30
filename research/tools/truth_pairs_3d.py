#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Truth pairs with a 3-D overlap requirement, not just feature matches.
#
# Why: feature matching (ORB + PnP) only proves two frames share SOME content.
# Two different corners of a hall can both see the same wall and match 300
# points while being metres apart. On 20261029_060258 the current truth pairs
# give IoU 0.119 - the two visits cover mostly different geometry, so calling
# them the same place is not supported.
#
# This tool adds the missing constraint: project B dense cloud into A frame
# with the PnP pose and require a high 3-D inlier ratio.
#
# Usage:
#     python research/tools/truth_pairs_3d.py --rec 20261029_060258 --out .tmp/truth3d.tsv
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nav_loop import LoopConfig, relative_pose  # noqa: E402
from research.tools.loop_replay import features_of       # noqa: E402


def dense(path, rng=5.0, cap=20000, seed=0):
    with np.load(path) as z:
        p = np.asarray(z["pts"], np.float32)
    ok = np.isfinite(p).all(1) & (np.hypot(p[:, 0], p[:, 1]) <= rng)
    p = p[ok]
    if len(p) > cap:
        idx = np.random.default_rng(seed).choice(len(p), cap, replace=False)
        p = p[idx]
    return p


def overlap_ratio(PA, PB, R_ab, t_ab, tol=0.25):
    if len(PA) < 500 or len(PB) < 500:
        return 0.0, 0.0
    P = PB.astype(np.float64) @ np.asarray(R_ab, np.float64).T + np.asarray(t_ab, np.float64)
    tree = cKDTree(PA.astype(np.float64))
    d, _ = tree.query(P)
    return float((d <= tol).mean()), float(np.median(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--min-inliers", type=int, default=200)
    ap.add_argument("--min-coverage", type=float, default=0.30)
    ap.add_argument("--min-overlap", type=float, default=0.45)
    ap.add_argument("--tol", type=float, default=0.25)
    args = ap.parse_args()

    rec = ROOT / "navmesh_recordings" / args.rec
    cfg = LoopConfig()

    cp = Path(args.pairs) if args.pairs else (ROOT / ".tmp" / ("truth_%s.tsv" % args.rec))
    if not cp.exists():
        raise SystemExit("[fail] missing candidates %s" % cp)
    pairs = []
    for line in cp.open(encoding="utf-8"):
        if line.startswith("a"):
            continue
        f = line.split()
        pairs.append((int(f[0]), int(f[1])))
    uniq = sorted({tuple(sorted(p)) for p in pairs})
    print("[cand ] %d candidate pairs" % len(uniq), flush=True)

    feats = {}
    clouds = {}
    def feat_of(k):
        if k not in feats:
            with np.load(rec / ("kf/%06d.npz" % k)) as z:
                feats[k] = features_of(z)
        return feats[k]
    def cloud_of(k):
        if k not in clouds:
            clouds[k] = dense(rec / ("kf/%06d.npz" % k))
        return clouds[k]

    kept = []
    print("%14s %8s %6s %8s %8s %s" % ("pair", "inliers", "cov", "overlap", "med_d", "keep"), flush=True)
    for a, b in uniq:
        try:
            fa, fb = feat_of(a), feat_of(b)
            PA, PB = cloud_of(a), cloud_of(b)
        except FileNotFoundError:
            continue
        rel, why = relative_pose(fa, fb, cfg)
        if rel is None:
            print("%14s  reject %s" % ("%d-%d" % (a, b), why), flush=True)
            continue
        if rel["inliers"] < args.min_inliers or rel["coverage"] < args.min_coverage:
            print("%14s %8d %6.2f  low quality" % ("%d-%d" % (a, b), rel["inliers"], rel["coverage"]), flush=True)
            continue
        ov, med = overlap_ratio(PA, PB, rel["R_ab"], rel["t_ab"], args.tol)
        ok = ov >= args.min_overlap
        kept.append((a, b, rel["inliers"], rel["coverage"], ov, med))
        print("%14s %8d %6.2f %8.3f %8.3f %s" % (
            "%d-%d" % (a, b), rel["inliers"], rel["coverage"], ov, med,
            "OK" if ok else "DROP"), flush=True)

    good = [p for p in kept if p[4] >= args.min_overlap]
    print()
    print("[3d   ] kept %d / %d (overlap >= %.2f)" % (len(good), len(kept), args.min_overlap), flush=True)
    if good:
        ov = np.array([p[4] for p in good])
        print("        overlap median %.3f  min %.3f  max %.3f" % (np.median(ov), ov.min(), ov.max()), flush=True)
    if args.out and good:
        out = Path(args.out) if Path(args.out).is_absolute() else (ROOT / args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            f.write("a\tb\tinliers\tcoverage\toverlap\tmed_dist\n")
            for p in good:
                f.write("%d\t%d\t%d\t%.3f\t%.3f\t%.3f\n" % p)
        print("[out  ] %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
