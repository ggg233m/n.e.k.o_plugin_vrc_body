# -*- coding: utf-8 -*-
"""Is the VRChat capture actually like the Middlebury test images?

Compare the two at the same width (720) on the statistics that decide whether
a stereo matcher succeeds: sharpness, gradient strength, how much of the frame
is textureless, contrast, and the disparity range.
"""
import os
import sys
import glob
import numpy as np
import cv2

ROOT = os.path.abspath(".")
RECS = ["20261010_033237", "20261013_223033", "20261014_003723"]
MB = os.path.join(ROOT, ".tmp", "mb2014", "ex")
W = 720


def stats(g):
    """All the numbers on one grey image."""
    g = g.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    # textureless = gradient below 5 grey levels per pixel (typical flat wall)
    flat = float((mag < 5.0).mean() * 100.0)
    return dict(
        mean=float(g.mean()),
        std=float(g.std()),
        lapvar=float(lap.var()),
        grad_p50=float(np.percentile(mag, 50)),
        grad_p90=float(np.percentile(mag, 90)),
        grad_p99=float(np.percentile(mag, 99)),
        flat_pct=flat,
        # fraction of pixels that carry real structure
        strong_pct=float((mag > 30.0).mean() * 100.0),
    )


def load_vrchat(rec, kf=0):
    p = os.path.join(ROOT, "navmesh_recordings", rec, "kf_img")
    for name in ("%06d_L.png" % kf, "%06d_L.jpg" % kf):
        f = os.path.join(p, name)
        if os.path.exists(f):
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if im is not None:
                return im
    return None


rows = []

print("=" * 100)
print("VRChat captures (native 720x405)")
print("=" * 100)
print("%-18s %-7s %-7s %-9s %-9s %-9s %-9s %-8s"
      % ("recording", "mean", "std", "lapvar", "grad_p50", "grad_p90", "flat%", "strong%"))
for rec in RECS:
    g = load_vrchat(rec, 0)
    if g is None:
        print("%-18s MISSING" % rec)
        continue
    s = stats(g)
    rows.append(("vrc:" + rec, s, g.shape))
    print("%-18s %-7.1f %-7.1f %-9.2f %-9.2f %-9.2f %-9.1f %-8.2f"
          % (rec, s["mean"], s["std"], s["lapvar"], s["grad_p50"],
             s["grad_p90"], s["flat_pct"], s["strong_pct"]))

print()
print("=" * 100)
print("Middlebury 2014, downscaled to the same 720 width")
print("=" * 100)
print("%-18s %-7s %-7s %-9s %-9s %-9s %-9s %-8s"
      % ("scene", "mean", "std", "lapvar", "grad_p50", "grad_p90", "flat%", "strong%"))
MBS = ["Adirondack", "Bicycle1", "Classroom1", "Couch", "Flowers",
       "Motorcycle", "Piano", "Playtable", "Recycle", "Shelves"]
mb_rows = []
for name in MBS:
    f = os.path.join(MB, name + "-perfect", "im0.png")
    if not os.path.exists(f):
        continue
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if im is None:
        continue
    h = int(round(im.shape[0] * W / float(im.shape[1])))
    g = cv2.resize(im, (W, h), interpolation=cv2.INTER_AREA)
    s = stats(g)
    mb_rows.append((name, s))
    print("%-18s %-7.1f %-7.1f %-9.2f %-9.2f %-9.2f %-9.1f %-8.2f"
          % (name, s["mean"], s["std"], s["lapvar"], s["grad_p50"],
             s["grad_p90"], s["flat_pct"], s["strong_pct"]))

print()
print("=" * 100)
print("Summary (median across scenes)")
print("=" * 100)
vrc = [r[1] for r in rows if r[0].startswith("vrc:")]
mbm = [r[1] for r in mb_rows]


def med(rs, k):
    return float(np.median([r[k] for r in rs]))


keys = [("std", "contrast (grey std)"), ("lapvar", "sharpness (Laplacian var)"),
        ("grad_p50", "gradient p50"), ("grad_p90", "gradient p90"),
        ("flat_pct", "textureless % (grad<5)"), ("strong_pct", "strong edges % (grad>30)")]
print("%-32s %-14s %-14s %-10s" % ("metric", "VRChat", "Middlebury", "ratio"))
print("-" * 72)
for k, label in keys:
    a, b = med(vrc, k), med(mbm, k)
    r = (a / b) if b else float("nan")
    print("%-32s %-14.3f %-14.3f %-10.2f" % (label, a, b, r))

print()
print("=" * 100)
print("Disparity range actually present")
print("=" * 100)
print("VRChat rig: b=0.126 m, f=202.5 px, obstacle band 2-8 m")
for z in (2.0, 3.0, 4.0, 6.0, 8.0):
    print("  z=%.1f m -> d = %.2f px" % (z, 25.515 / z))
print()
print("Middlebury at 720 wide (median GT disparity):")
sys.path.insert(0, ROOT)
from research.tools.mb_color_res import load_scene, resize_pair
for name in MBS:
    sc = load_scene(name)
    if sc is None:
        continue
    _L, _R, gt = resize_pair(sc[0], sc[1], sc[2], W)
    v = gt[np.isfinite(gt)]
    print("  %-12s d_p50=%6.1f  d_p90=%6.1f  ratio to rig@4m: %5.1fx"
          % (name, float(np.median(v)), float(np.percentile(v, 90)),
             float(np.median(v)) / 6.38))