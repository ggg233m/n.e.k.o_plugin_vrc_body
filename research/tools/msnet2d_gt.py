# -*- coding: utf-8 -*-
"""MobileStereoNet 2D with ground truth.

1.887 M params, the "lightweight backbone" class the earlier round never tested.
maxdisp must match the checkpoint; try 192 (the standard SceneFlow setting) and
fall back if the tensor shapes disagree.
"""
import os, sys, time, numpy as np, cv2, torch

ROOT = os.path.abspath(".")
sys.path.insert(0, ROOT)
SRC = os.path.join(ROOT, ".tmp", "msnet_src")
sys.path.insert(0, SRC)

from models.MSNet2D import MSNet2D
from research.tools.mb_color_res import load_scene, resize_pair, sgbm_for, gt_metrics, SCENES

CKPT = os.path.join(ROOT, ".tmp", "stereo_models", "msnet", "msnet2d_sf.ckpt")
DEV = torch.device("xpu")

state = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
state = dict((k.replace("module.", "", 1), v) for k, v in state.items())

built = None
for md in (192, 256, 128, 96):
    try:
        net = MSNet2D(md)
        missing, unexpected = net.load_state_dict(state, strict=False)
        if not unexpected and not missing:
            built = (net, md)
            print("[ok] maxdisp=%d  strict load clean" % md)
            break
        print("  maxdisp=%-4d missing=%d unexpected=%d" % (md, len(missing), len(unexpected)))
    except Exception as exc:
        print("  maxdisp=%-4d build failed: %s" % (md, str(exc)[:70]))

if built is None:
    print("[fail] no matching maxdisp")
    raise SystemExit(1)

net, maxdisp = built
net = net.eval().to(DEV)

IM_MEAN = 0.485
IM_STD = 0.229


def prep(gray):
    t = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]
    t = t.repeat(1, 3, 1, 1)
    t = (t - IM_MEAN) / IM_STD
    return t.to(DEV)


def infer(g_l, g_r):
    x1, x2 = prep(g_l), prep(g_r)
    with torch.no_grad():
        out = net(x1, x2)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    return out.detach().float().cpu().numpy()[0]


for mw, mh in ((640, 480), (320, 240)):
    print()
    print("=" * 88)
    print("MSNet2D @ %dx%d (maxdisp=%d)" % (mw, mh, maxdisp))
    print("=" * 88)
    print("%-12s %-9s %-9s %-9s %-9s" % ("scene", "ms", "EPE", "bad1%", "cov%"))
    rows = []
    ts = []
    for name in SCENES:
        sc = load_scene(name)
        if sc is None:
            continue
        L, R, G = sc
        Ls, Rs, gt = resize_pair(L, R, G, mw)
        if Ls.shape[0] != mh:
            Ls = cv2.resize(Ls, (mw, mh), interpolation=cv2.INTER_AREA)
            Rs = cv2.resize(Rs, (mw, mh), interpolation=cv2.INTER_AREA)
            g2 = cv2.resize(np.nan_to_num(gt, posinf=0.0), (mw, mh), interpolation=cv2.INTER_NEAREST)
            gt = np.where(g2 > 0, g2, np.inf).astype(np.float32)
        gl = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
        infer(gl, gr)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        d = infer(gl, gr)
        torch.xpu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
        mm = gt_metrics(d, gt)
        rows.append(mm)
        print("%-12s %-9.1f %-9.3f %-9.2f %-9.1f"
              % (name, ts[-1], mm["epe"], mm["bad1"], mm["cov"]))
    print("-" * 88)
    print("%-12s %-9.1f %-9.3f %-9.2f %-9.1f"
          % ("MEAN", float(np.median(ts)),
             float(np.mean([r["epe"] for r in rows])),
             float(np.mean([r["bad1"] for r in rows])),
             float(np.mean([r["cov"] for r in rows]))))
    print("  (SGBM at this grid: see mb_sgbm.log ; CREStereo@640x480: EPE 0.327 bad1 4.79)")