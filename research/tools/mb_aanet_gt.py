# -*- coding: utf-8 -*-
"""AANet / StereoNet-AA with GROUND TRUTH -- the gap I left open.

Every AANet number in this repo was scored with the photometric residual, and
that metric is anti-correlated across resolutions.  These two are precisely the
lightweight backbones that could be both fast and accurate, so their accuracy
was never actually established.  Measure it now, same grid as CREStereo.
"""
import os
import sys
import time
import numpy as np
import cv2
import torch

ROOT = os.path.abspath(".")
sys.path.insert(0, ROOT)
from research.tools.mb_color_res import (load_scene, resize_pair, sgbm_for,
                                         gt_metrics, build_torch, to_tensor,
                                         as_gray3, MODELS, SCENES)

ALIGN = dict((m[0], m[3]) for m in MODELS)


def prep_pair(name, mw, mh):
    sc = load_scene(name)
    if sc is None:
        return None
    L, R, G = sc
    Ls, Rs, gt = resize_pair(L, R, G, mw)
    if Ls.shape[0] != mh:
        Ls = cv2.resize(Ls, (mw, mh), interpolation=cv2.INTER_AREA)
        Rs = cv2.resize(Rs, (mw, mh), interpolation=cv2.INTER_AREA)
        g2 = cv2.resize(np.nan_to_num(gt, posinf=0.0), (mw, mh),
                        interpolation=cv2.INTER_NEAREST)
        gt = np.where(g2 > 0, g2, np.inf).astype(np.float32)
    fin = np.isfinite(gt)
    over = float((gt[fin] > 192).mean()) if fin.any() else 0.0
    return Ls, Rs, gt, over


models = {}
for nm, ck, cfg, _a in MODELS:
    try:
        models[nm] = build_torch(torch, ck, cfg)
        print("[ok]", nm)
    except Exception as exc:
        print("[skip]", nm, str(exc)[:80])

for mw, mh in ((640, 480), (320, 240)):
    print()
    print("=" * 92)
    print("grid %dx%d : AANet family vs SGBM, dense ground truth" % (mw, mh))
    print("=" * 92)
    print("%-13s %-6s %-8s %-9s %-8s %-8s" % ("model", "input", "ms", "EPE", "bad1%", "cov%"))
    for nm in sorted(models):
        for tag in ("gray3", "rgb"):
            rows = []
            ts = []
            for name in SCENES:
                pp = prep_pair(name, mw, mh)
                if pp is None:
                    continue
                Ls, Rs, gt, over = pp
                A = Ls if tag == "rgb" else as_gray3(Ls)
                B = Rs if tag == "rgb" else as_gray3(Rs)
                x1 = to_tensor(torch, A, ALIGN[nm])
                x2 = to_tensor(torch, B, ALIGN[nm])
                with torch.no_grad():
                    models[nm](x1, x2)
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = models[nm](x1, x2)
                torch.xpu.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
                disp = out[-1].detach().float().cpu().numpy()[0]
                disp = disp[:gt.shape[0], :gt.shape[1]]
                rows.append(gt_metrics(disp, gt))
            if rows:
                print("%-13s %-6s %-8.1f %-9.3f %-8.2f %-8.1f"
                      % (nm, tag, float(np.median(ts)),
                         float(np.mean([r["epe"] for r in rows])),
                         float(np.mean([r["bad1"] for r in rows])),
                         float(np.mean([r["cov"] for r in rows]))))

    # SGBM reference on the identical grid
    rows = []
    for name in SCENES:
        pp = prep_pair(name, mw, mh)
        if pp is None:
            continue
        Ls, Rs, gt, over = pp
        m, nd = sgbm_for(gt)
        lg = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        d = m.compute(cv2.cvtColor(lg, cv2.COLOR_GRAY2BGR),
                      cv2.cvtColor(rg, cv2.COLOR_GRAY2BGR)).astype(np.float32) / 16.0
        ms = (time.perf_counter() - t0) * 1e3
        mm = gt_metrics(d, gt)
        mm["ms"] = ms
        rows.append(mm)
    print("%-13s %-6s %-8.1f %-9.3f %-8.2f %-8.1f"
          % ("SGBM", "gray3", float(np.median([r["ms"] for r in rows])),
             float(np.mean([r["epe"] for r in rows])),
             float(np.mean([r["bad1"] for r in rows])),
             float(np.mean([r["cov"] for r in rows]))))