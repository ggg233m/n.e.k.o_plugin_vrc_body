# -*- coding: utf-8 -*-
"""FFS on Middlebury with ground truth, true RGB vs grey-replicated.

Preprocessing follows the earlier .tmp/_final_ffs.py: ImageNet mean/std in the
0..255 domain and RGB channel order (the ONNX was exported from the FFS repo).
That earlier script fed grey replicated and scored with the photometric
residual; this one has ground truth and real colour.
"""
import os, sys, time, numpy as np, cv2
sys.path.insert(0, os.path.abspath("."))
from research.tools.mb_color_res import load_scene, resize_pair, sgbm_for, gt_metrics, SCENES
import openvino as ov

MODELS = os.path.join(os.path.abspath("."), ".tmp", "stereo_models", "ffs")
MEAN = np.array([123.675, 116.28, 103.53], np.float32).reshape(1, 3, 1, 1)
STD = np.array([58.395, 57.12, 57.375], np.float32).reshape(1, 3, 1, 1)


def prep_rgb(bgr, h, w):
    x = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)
    t = rgb.transpose(2, 0, 1)[None]
    return (t - MEAN) / STD


def prep_gray3(bgr, h, w):
    x = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t = np.stack([g] * 3, axis=0)[None]
    return (t - MEAN) / STD


def main():
    core = ov.Core()
    if "GPU" not in core.available_devices:
        print("[skip] no GPU"); return 2
    p = os.path.join(MODELS, "c_ffs_dynhw.onnx")
    if not os.path.exists(p):
        print("[skip] missing", p); return 2

    for h, w in ((480, 736), (384, 640)):
        m = core.read_model(p)
        m.reshape({m.input(0): [1, 3, h, w], m.input(1): [1, 3, h, w]})
        cm = core.compile_model(m, "GPU")
        for prep, iname in ((prep_rgb, "rgb"), (prep_gray3, "gray3")):
            print()
            print("=" * 86)
            print("FFS %dx%d input=%-5s metric vs dense GT" % (w, h, iname))
            print("=" * 86)
            print("%-12s %-8s %-8s %-8s %-8s" % ("scene", "ms", "cov%", "EPE", "bad1%"))
            rows = []
            for name in SCENES:
                sc = load_scene(name)
                if sc is None:
                    continue
                L, R, G = sc
                Ls, Rs, gt0 = resize_pair(L, R, G, w)
                if Ls.shape[0] != h:
                    Ls = cv2.resize(Ls, (w, h), interpolation=cv2.INTER_AREA)
                    Rs = cv2.resize(Rs, (w, h), interpolation=cv2.INTER_AREA)
                    g2 = cv2.resize(np.nan_to_num(gt0, posinf=0.0), (w, h),
                                    interpolation=cv2.INTER_NEAREST)
                    gt = np.where(g2 > 0, g2, np.inf).astype(np.float32)
                else:
                    gt = gt0
                x1, x2 = prep(Ls, h, w), prep(Rs, h, w)
                cm([x1, x2])
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    o = cm([x1, x2])[cm.output(0)]
                    ts.append((time.perf_counter() - t0) * 1e3)
                d = np.abs(o[0, 0]).astype(np.float32)
                mm = gt_metrics(d, gt)
                rows.append(mm)
                print("%-12s %-8.1f %-8.1f %-8.3f %-8.2f"
                      % (name, float(np.median(ts)), mm["cov"], mm["epe"], mm["bad1"]))
            print("-" * 86)
            print("%-12s %-8s %-8.1f %-8.3f %-8.2f  (mean)"
                  % ("MEAN", "-", float(np.mean([r["cov"] for r in rows])),
                     float(np.mean([r["epe"] for r in rows])),
                     float(np.mean([r["bad1"] for r in rows]))))

        # SGBM at the same grid
        rows = []
        for name in SCENES:
            sc = load_scene(name)
            if sc is None:
                continue
            L, R, G = sc
            Ls, Rs, gt = resize_pair(L, R, G, w)
            if Ls.shape[0] != h:
                Ls = cv2.resize(Ls, (w, h), interpolation=cv2.INTER_AREA)
                Rs = cv2.resize(Rs, (w, h), interpolation=cv2.INTER_AREA)
                g2 = cv2.resize(np.nan_to_num(gt, posinf=0.0), (w, h),
                                interpolation=cv2.INTER_NEAREST)
                gt = np.where(g2 > 0, g2, np.inf).astype(np.float32)
            m2, nd = sgbm_for(gt)
            lg = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY)
            rg = cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
            d = m2.compute(cv2.cvtColor(lg, cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(rg, cv2.COLOR_GRAY2BGR)).astype(np.float32) / 16.0
            rows.append(gt_metrics(d, gt))
        print("-" * 86)
        print("%-12s %-8s %-8.1f %-8.3f %-8.2f  (SGBM gray3 ref @ %dx%d)"
              % ("SGBM", "-", float(np.mean([r["cov"] for r in rows])),
                 float(np.mean([r["epe"] for r in rows])),
                 float(np.mean([r["bad1"] for r in rows])), w, h))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())