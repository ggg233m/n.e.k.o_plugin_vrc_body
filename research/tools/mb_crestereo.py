# -*- coding: utf-8 -*-
"""CREStereo on Middlebury 2014, with GROUND TRUTH and TRUE COLOUR.

The existing crestereo_bench.py only ran at one scale on grey-replicated input
and scored with the photometric residual.  That residual is now known to be
anti-correlated across resolutions, so those numbers cannot rank CREStereo
against SGBM at a different scale.  This reruns it properly:

  * dense GT disparity from disp0.pfm
  * true RGB (the ONNX takes 3 channels; upstream feeds RGB, not grey)
  * several resolutions, each native to the model size the ONNX declares

The combined ONNX has 4 inputs: left/right at half size then left/right at
full size.  Output disparity is in pixels of the model input width.

Usage:  .venv/Scripts/python.exe -u research/tools/mb_crestereo.py
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.tools.mb_color_res import (load_scene, resize_pair, sgbm_for,
                                         gt_metrics, SCENES)

MODELS = os.path.join(ROOT, ".tmp", "stereo_models")
#: (subdir, tag, model input h, w).  The ONNX files are FIXED-shape.
ARMS = (("x10", "iter10", 720, 1280),)


def tensor_rgb(bgr, h, w):
    """BGR uint8 -> NCHW float32 RGB at (h, w), no normalisation.

    CREStereo upstream feeds raw 0..255 RGB.  Feeding grey-replicated RGB is
    what the old bench did; the models were trained on real colour, so we test
    both to keep the comparison honest.
    """
    x = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)
    return rgb.transpose(2, 0, 1)[None]


def tensor_gray3(bgr, h, w):
    x = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return np.stack([g] * 3, axis=0)[None].astype(np.float32)


def run(compiled, left, right, h, w, maker):
    lh, lw = h // 2, w // 2
    o = compiled([maker(left, lh, lw), maker(right, lh, lw),
                  maker(left, h, w), maker(right, h, w)])[compiled.output(0)]
    return np.abs(o[0, 0]).astype(np.float32)


def main():
    import openvino as ov

    core = ov.Core()
    if "GPU" not in core.available_devices:
        print("[skip] no GPU:", core.available_devices)
        return 2

    arms = {}
    for sub, tag, h, w in ARMS:
        p = os.path.join(MODELS, sub, "crestereo_combined_%s_%dx%d.onnx" % (tag, h, w))
        if not os.path.exists(p):
            print("[skip] missing", p)
            continue
        arms[tag] = (core.compile_model(core.read_model(p), "GPU"), h, w)
        print("[ok] loaded %s %dx%d" % (tag, h, w))
    if not arms:
        return 2

    for tag, (model, mh, mw) in arms.items():
        for maker, iname in ((tensor_rgb, "rgb"), (tensor_gray3, "gray3")):
            print()
            print("=" * 104)
            print("CREStereo %s @ %dx%d   input=%s   metric vs dense GT" % (tag, mw, mh, iname))
            print("=" * 104)
            print("%-12s %-7s %-9s %-8s %-8s %-8s" %
                  ("scene", "scale", "ms", "cov%", "EPE", "bad1%"))
            agg = []
            for name in SCENES:
                sc = load_scene(name)
                if sc is None:
                    continue
                L, R, G = sc
                # GT at the model input width so no rescaling of the
                # prediction is needed
                Ls, Rs, gt = resize_pair(L, R, G, mw)
                # the model wants h=mh; resize_pair gives mh only if aspect
                # matches, so force it and accept a tiny vertical stretch
                if Ls.shape[0] != mh:
                    Ls = cv2.resize(Ls, (mw, mh), interpolation=cv2.INTER_AREA)
                    Rs = cv2.resize(Rs, (mw, mh), interpolation=cv2.INTER_AREA)
                    gt = cv2.resize(np.nan_to_num(gt, posinf=0.0), (mw, mh),
                                    interpolation=cv2.INTER_NEAREST)
                    gt = np.where(gt > 0, gt, np.inf).astype(np.float32)

                run(model, Ls, Rs, mh, mw, maker)  # warmup
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    d = run(model, Ls, Rs, mh, mw, maker)
                    ts.append((time.perf_counter() - t0) * 1e3)
                mm = gt_metrics(d, gt)
                agg.append(mm)
                print("%-12s %-7d %-9.1f %-8.1f %-8.3f %-8.2f"
                      % (name, mw, float(np.median(ts)), mm["cov"],
                         mm["epe"], mm["bad1"]))
            if agg:
                print("-" * 104)
                print("%-12s %-7d %-9s %-8.1f %-8.3f %-8.2f  (mean over %d scenes)"
                      % ("MEAN", mw, "-",
                         float(np.mean([a["cov"] for a in agg])),
                         float(np.mean([a["epe"] for a in agg])),
                         float(np.mean([a["bad1"] for a in agg])), len(agg)))

    # SGBM reference at the same width
    print()
    print("=" * 104)
    print("SGBM reference at the same width (from the same GT pipeline)")
    print("=" * 104)
    for iname in ("gray3",):
        rows = []
        for name in SCENES:
            sc = load_scene(name)
            if sc is None:
                continue
            L, R, G = sc
            Ls, Rs, gt = resize_pair(L, R, G, 1280)
            m, nd = sgbm_for(gt)
            lg = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY)
            rg = cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
            d = m.compute(cv2.cvtColor(lg, cv2.COLOR_GRAY2BGR),
                          cv2.cvtColor(rg, cv2.COLOR_GRAY2BGR)).astype(np.float32) / 16.0
            rows.append(gt_metrics(d, gt))
        print("%-12s %-7d %-9s %-8.1f %-8.3f %-8.2f  (mean over %d scenes)"
              % ("SGBM " + iname, 1280, "-",
                 float(np.mean([r["cov"] for r in rows])),
                 float(np.mean([r["epe"] for r in rows])),
                 float(np.mean([r["bad1"] for r in rows])), len(rows)))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())