# -*- coding: utf-8 -*-
"""SGBM accuracy at rig-native long sides: 720, 1080, 1440.

Key design point: the Middlebury native width is 2880, so downscaling to 1080
is a genuine downsample (no upsampling artefacts).  Ground truth scaled with it.

Report both:
  - absolute EPE in disparity pixels (what the matcher achieves)
  - relZ median (what the map cares about), and the depth noise proxy
"""
import os
import sys
import time
import numpy as np
import cv2

ROOT = os.path.abspath(".")
sys.path.insert(0, ROOT)
from research.tools.mb_color_res import load_scene, resize_pair, sgbm_for, gt_metrics, SCENES

LADDER = [(720, 405), (1080, 608), (1440, 810), (2880, 1620)]

print("=" * 104)
print("SGBM accuracy by rig-native long side (Middlebury downscaled, dense GT)")
print("=" * 104)
print("%-12s %-9s %-10s %-9s %-9s %-10s %-12s" %
      ("long side", "size", "ms", "cov%", "EPE", "bad1%", "relZ med%"))
print("-" * 104)

store = {}
for W, H in LADDER:
    rows = []
    t_all = []
    for name in SCENES:
        sc = load_scene(name)
        if sc is None:
            continue
        L, R, G = sc
        Ls, Rs, gt = resize_pair(L, R, G, W)
        # force the exact 16:9 height the rig would capture
        if Ls.shape[0] != H:
            Ls = cv2.resize(Ls, (W, H), interpolation=cv2.INTER_AREA)
            Rs = cv2.resize(Rs, (W, H), interpolation=cv2.INTER_AREA)
            g2 = cv2.resize(np.nan_to_num(gt, posinf=0.0), (W, H), interpolation=cv2.INTER_AREA)
            ok = cv2.resize((np.isfinite(gt)).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
            gt = np.where(ok > 0.999, g2, np.inf).astype(np.float32)
        m, nd = sgbm_for(gt)
        lg = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
        m.compute(lg, rg)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            d = m.compute(lg, rg).astype(np.float32) / 16.0
            ts.append((time.perf_counter() - t0) * 1e3)
        t_all.append(float(np.median(ts)))
        mm = gt_metrics(d, gt)
        ok2 = np.isfinite(gt) & (d > 1.0)
        if ok2.sum() > 500:
            mm["relZ"] = float(np.median(np.abs(d[ok2] / gt[ok2] - 1.0)) * 100.0)
        else:
            mm["relZ"] = float("nan")
        mm["scene"] = name
        rows.append(mm)
    if not rows:
        continue
    store[W] = rows
    print("%-12d %-9s %-10.1f %-9.1f %-9.3f %-10.2f %-12.2f"
          % (W, "%dx%d" % (W, H), float(np.median(t_all)),
             float(np.mean([r["cov"] for r in rows])),
             float(np.mean([r["epe"] for r in rows])),
             float(np.mean([r["bad1"] for r in rows])),
             float(np.nanmean([r["relZ"] for r in rows]))))

print()
print("=" * 104)
print("Same, restricted to the 8-13 px disparity band nearest the rig")
print("=" * 104)
print("%-12s %-9s %-9s %-9s %-11s" % ("long side", "n_px", "EPE", "relZ%", "vs 720"))
print("-" * 60)
base = None
for W, H in LADDER:
    tot_e, tot_r, tot_n = [], [], 0
    for name in SCENES:
        sc = load_scene(name)
        if sc is None:
            continue
        L, R, G = sc
        Ls, Rs, gt = resize_pair(L, R, G, W)
        if Ls.shape[0] != H:
            Ls = cv2.resize(Ls, (W, H), interpolation=cv2.INTER_AREA)
            Rs = cv2.resize(Rs, (W, H), interpolation=cv2.INTER_AREA)
            g2 = cv2.resize(np.nan_to_num(gt, posinf=0.0), (W, H), interpolation=cv2.INTER_AREA)
            ok = cv2.resize((np.isfinite(gt)).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
            gt = np.where(ok > 0.999, g2, np.inf).astype(np.float32)
        m, nd = sgbm_for(gt)
        lg = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
        d = m.compute(lg, rg).astype(np.float32) / 16.0
        sel = np.isfinite(gt) & (gt >= 8) & (gt < 13) & (d > 1.0)
        if sel.sum() > 300:
            tot_e.append(float(np.abs(d[sel] - gt[sel]).mean()))
            tot_r.append(float(np.median(np.abs(d[sel] / gt[sel] - 1.0)) * 100))
            tot_n += int(sel.sum())
    if tot_e:
        e = float(np.mean(tot_e))
        r = float(np.mean(tot_r))
        if base is None:
            base = e
        print("%-12d %-9d %-9.3f %-9.2f %-11.2fx" % (W, tot_n, e, r, base / e))