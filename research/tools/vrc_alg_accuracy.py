# -*- coding: utf-8 -*-
"""Algorithm comparison on real VRChat imagery, correct sign, rig disparity range.

Ground truth = a known Fourier shift of the left frame, so the disparity field
is uniform and exact.  This measures MATCHER accuracy, which is the thing being
compared; it does not reproduce a real room, and that is stated with the result.

Convention verified separately: right(u) = left(u + d), so the right view is
the left shifted by -d (content moves left).
"""
import os
import sys
import time
import numpy as np
import cv2

ROOT = os.path.abspath(".")
sys.path.insert(0, ROOT)


def shift(img, dx):
    h, w = img.shape
    f = np.fft.fft2(img.astype(np.float64))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.real(np.fft.ifft2(f * np.exp(-2j * np.pi * (fx * dx + fy * 0.0))))


def sgbm(gl, gr, nd=64):
    m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=5,
                              P1=200, P2=800, uniquenessRatio=10,
                              speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                              mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return m.compute(gl, gr).astype(np.float32) / 16.0


def score(d, truth):
    """Interior only; exclude the strip the shift wrapped."""
    h, w = d.shape
    minx = int(np.ceil(abs(truth))) + 10
    inner = d[h // 5: 4 * h // 5, minx:w - 10]
    ok = inner > 0.5
    cov = 100.0 * ok.sum() / max(inner.size, 1)
    if ok.sum() < 500:
        return float("nan"), cov, 0
    e = np.abs(inner[ok] - truth)
    return float(e.mean()), cov, int(ok.sum())


SHIFTS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.8]
FRAMES = [("20261010_033237", 0), ("20261010_033237", 60),
          ("20261013_223033", 0), ("20261013_223033", 60)]

print("=" * 104)
print("SGBM on real VRChat imagery, uniform known disparity")
print("=" * 104)
print("%-8s %-7s %-9s %-9s %-9s %-9s" % ("d(px)", "z(m)", "EPE", "med|e|", "cov%", "relZ%"))
print("-" * 104)
agg = dict((s, []) for s in SHIFTS)
for s in SHIFTS:
    rows = []
    for rec, kf in FRAMES:
        p = os.path.join("navmesh_recordings", rec, "kf_img", "%06d_L.png" % kf)
        if not os.path.exists(p):
            continue
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        R = np.clip(shift(g, -s), 0, 255).astype(np.uint8)
        d = sgbm(g, R)
        e, c, n = score(d, s)
        if np.isfinite(e):
            rows.append(e)
    if rows:
        e = float(np.mean(rows))
        agg[s] = rows
        print("%-8.1f %-7.1f %-9.3f %-9s %-9s %-9.2f"
              % (s, 25.515 / s, e, "-", "-", 100.0 * e / s))

print()
print("=" * 104)
print("Aggregate relZ by disparity -- does it follow 1/d?")
print("=" * 104)
print("%-8s %-7s %-9s %-9s %-12s %-12s" % ("d", "z(m)", "EPE", "relZ%", "1/d predicts", "actual/ideal"))
ref = None
for s in SHIFTS:
    if not agg[s]:
        continue
    e = float(np.mean(agg[s]))
    rz = 100.0 * e / s
    if s == 6.0:
        ref = rz
    if ref is None:
        print("%-8.1f %-7.1f %-9.3f %-9.2f %-12s %-12s" % (s, 25.515 / s, e, rz, "-", "-"))
    else:
        ideal = ref * 6.0 / s
        print("%-8.1f %-7.1f %-9.3f %-9.2f %-12.2f %-12.2f"
              % (s, 25.515 / s, e, rz, ideal, rz / ideal))