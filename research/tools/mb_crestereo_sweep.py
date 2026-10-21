# -*- coding: utf-8 -*-
"""CREStereo resolution sweep with ground truth, excluding truncated scenes."""
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
ARMS = (("x10", "iter10", 720, 1280),
        ("x10", "iter10", 480, 640),
        ("x10", "iter10", 360, 640),
        ("x2", "iter2", 480, 640))
CEIL = 192.0


def tensor_rgb(bgr, h, w):
    x = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)
    return rgb.transpose(2, 0, 1)[None]


def tensor_gray3(bgr, h, w):
    x = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    return np.stack([g] * 3, axis=0)[None].astype(np.float32)


def run(compiled, left, right, h, w, maker):
    lh, lw = h // 2, w // 2
    o = compiled([maker(left, lh, lw), maker(right, lh, lw),
                  maker(left, h, w), maker(right, h, w)])[compiled.output(0)]
    return np.abs(o[0, 0]).astype(np.float32)


def prep_pair(name, mw, mh):
    sc = load_scene(name)
    if sc is None:
        return None
    L, R, G = sc
    Ls, Rs, gt = resize_pair(L, R, G, mw)
    if Ls.shape[0] != mh:
        Ls = cv2.resize(Ls, (mw, mh), interpolation=cv2.INTER_AREA)
        Rs = cv2.resize(Rs, (mw, mh), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(np.nan_to_num(gt, posinf=0.0), (mw, mh),
                        interpolation=cv2.INTER_NEAREST)
        gt = np.where(gt > 0, gt, np.inf).astype(np.float32)
    fin = np.isfinite(gt)
    over = float((gt[fin] > CEIL).mean()) if fin.any() else 0.0
    return Ls, Rs, gt, over


def main():
    import openvino as ov
    core = ov.Core()
    if "GPU" not in core.available_devices:
        print("[skip] no GPU")
        return 2
    arms = {}
    for sub, tag, h, w in ARMS:
        p = os.path.join(MODELS, sub, "crestereo_combined_%s_%dx%d.onnx" % (tag, h, w))
        if not os.path.exists(p):
            print("[skip] missing", os.path.basename(p))
            continue
        arms[(tag, w, h)] = core.compile_model(core.read_model(p), "GPU")
    print("[ok] arms:", sorted(arms))

    for (tag, mw, mh), model in sorted(arms.items()):
        for maker, iname in ((tensor_rgb, "rgb"), (tensor_gray3, "gray3")):
            rows = []
            for name in SCENES:
                pp = prep_pair(name, mw, mh)
                if pp is None:
                    continue
                Ls, Rs, gt, over = pp
                run(model, Ls, Rs, mh, mw, maker)
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    d = run(model, Ls, Rs, mh, mw, maker)
                    ts.append((time.perf_counter() - t0) * 1e3)
                mm = gt_metrics(d, gt)
                mm["over"] = over
                mm["scene"] = name
                mm["ms"] = float(np.median(ts))
                rows.append(mm)
            clean = [r for r in rows if r["over"] < 0.01]
            excl = ",".join(r["scene"] for r in rows if r["over"] >= 0.01) or "none"
            print()
            print("%s %dx%d input=%-5s  ms=%-7.1f  cov=%.1f%%"
                  % (tag, mw, mh, iname,
                     float(np.median([r["ms"] for r in rows])),
                     float(np.mean([r["cov"] for r in rows]))))
            print("  all   n=%-2d EPE=%-8.3f bad1=%-6.2f"
                  % (len(rows), float(np.mean([r["epe"] for r in rows])),
                     float(np.mean([r["bad1"] for r in rows]))))
            print("  clean n=%-2d EPE=%-8.3f bad1=%-6.2f   excluded: %s"
                  % (len(clean), float(np.mean([r["epe"] for r in clean])),
                     float(np.mean([r["bad1"] for r in clean])), excl))

    print()
    print("=" * 96)
    print("SGBM gray3 at the same grid widths")
    print("=" * 96)
    for mw, mh in ((1280, 720), (640, 480)):
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
            mm["over"] = over
            mm["scene"] = name
            mm["ms"] = ms
            rows.append(mm)
        clean = [r for r in rows if r["over"] < 0.01]
        excl = ",".join(r["scene"] for r in rows if r["over"] >= 0.01) or "none"
        print("SGBM %dx%d ms=%6.1f | all n=%d EPE=%.3f | clean n=%d EPE=%.3f (excl %s)"
              % (mw, mh, float(np.median([r["ms"] for r in rows])), len(rows),
                 float(np.mean([r["epe"] for r in rows])), len(clean),
                 float(np.mean([r["epe"] for r in clean])), excl))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())