# -*- coding: utf-8 -*-
"""Resolution vs baseline: which buys more depth accuracy per unit of d?

Both levers act through d = f*b/z, and relative depth error is
    dz/z = dd/d
so the question is whether dd (the disparity error in pixels) grows when you
use the lever.  Hypothesis: it grows with resolution (finer sampling, more
aliasing) but not with baseline (same images, just more parallax).

Measured here: dd as a function of the disparity itself, by binning Middlebury
pixels by GT disparity.  That curve then predicts what a baseline increase
would do, and the resolution sweep already measured the other lever.
"""
import os
import sys
import numpy as np
import cv2

ROOT = os.path.abspath(".")
sys.path.insert(0, ROOT)
from research.tools.mb_color_res import load_scene, resize_pair, sgbm_for, SCENES


def dd_by_disparity(width, bins):
    """Median |d_est - d_gt| per GT-disparity bin, pooled over scenes."""
    acc = dict((i, []) for i in range(len(bins) - 1))
    for name in SCENES:
        sc = load_scene(name)
        if sc is None:
            continue
        L, R, G = sc
        Ls, Rs, gt = resize_pair(L, R, G, width)
        m, nd = sgbm_for(gt)
        lg = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
        d = m.compute(cv2.cvtColor(lg, cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(rg, cv2.COLOR_GRAY2BGR)).astype(np.float32) / 16.0
        ok = np.isfinite(gt) & (d > 1.0)
        e = np.abs(d[ok] - gt[ok])
        g = gt[ok]
        for i in range(len(bins) - 1):
            sel = (g >= bins[i]) & (g < bins[i + 1])
            if sel.sum() > 2000:
                acc[i].append(e[sel])
    out = []
    for i in range(len(bins) - 1):
        if not acc[i]:
            out.append((0.5 * (bins[i] + bins[i + 1]), float("nan"), 0))
            continue
        v = np.concatenate(acc[i])
        out.append((0.5 * (bins[i] + bins[i + 1]), float(np.median(v)), int(v.size)))
    return out


print("=" * 92)
print("dd (median |error| in px) as a function of the disparity itself")
print("=" * 92)
for width in (720, 1440):
    bins = [4, 8, 16, 32, 64, 128, 256]
    rows = dd_by_disparity(width, bins)
    print()
    print("W=%d" % width)
    print("  %-10s %-12s %-12s %-12s" % ("d bin", "d centre", "median dd", "relZ %"))
    for centre, dd, n in rows:
        if not np.isfinite(dd):
            print("  %-10s %-12.1f %-12s %-12s" % ("-", centre, "n/a", "n/a"))
            continue
        print("  %-10s %-12.1f %-12.3f %-12.2f" % ("%d-%d" % (centre * 0.667, centre * 1.333),
                                                   centre, dd, 100.0 * dd / centre))

print()
print("=" * 92)
print("Predicted relZ under baseline scaling (d scales, dd read off the W=720 curve)")
print("=" * 92)
bins = [4, 8, 16, 32, 64, 128, 256]
rows = dd_by_disparity(720, bins)
valid = [(c, dd) for c, dd, n in rows if np.isfinite(dd)]


def dd_at(d):
    """Piecewise-linear interpolation of the measured error curve."""
    xs = np.array([c for c, _ in valid], float)
    ys = np.array([dd for _, dd in valid], float)
    return float(np.interp(d, xs, ys))


d0 = 27.0  # typical Middlebury @720 median
print("%-10s %-10s %-12s %-12s %-10s" % ("baseline", "d_med", "dd", "relZ %", "vs 1x"))
base = None
for k in (1.0, 1.5, 2.0, 3.0, 4.0):
    dk = d0 * k
    ddk = dd_at(dk)
    rz = 100.0 * ddk / dk
    if base is None:
        base = rz
    print("%-10.1fx %-10.1f %-12.3f %-12.2f %-10.2f" % (k, dk, ddk, rz, rz / base))

print()
print("Measured relZ under resolution scaling (from mb_relres.log, same pipeline):")
for w, rz in ((360, 1.68), (480, 1.30), (720, 0.91), (960, 0.69), (1440, 0.53), (2880, 0.40)):
    print("  W=%-5d relZ=%.2f%%   d_med ~ %.1f px" % (w, rz, d0 * w / 720.0))