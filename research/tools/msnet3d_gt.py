# -*- coding: utf-8 -*-
"""MobileStereoNet 3D with ground truth (SF checkpoint is the 3D variant)."""
import os, sys, time, numpy as np, cv2, torch

ROOT = os.path.abspath(".")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, ".tmp", "msnet_src"))

from models.MSNet3D import MSNet3D
from research.tools.mb_color_res import load_scene, resize_pair, gt_metrics, SCENES

CK = os.path.join(ROOT, ".tmp", "stereo_models", "msnet", "msnet2d_sf.ckpt")
DEV = torch.device("xpu")

state = torch.load(CK, map_location="cpu", weights_only=False)["model"]
state = dict((k.replace("module.", "", 1), v) for k, v in state.items())

net = None
maxdisp = None
for md in (192, 256, 128):
    try:
        n = MSNet3D(md)
        r = n.load_state_dict(state, strict=False)
        nmiss = len(r.missing_keys)
        nune = len(r.unexpected_keys)
        if nmiss == 0 and nune == 0:
            net, maxdisp = n, md
            print("[ok] MSNet3D maxdisp=%d strict-clean" % md)
            break
        print("  maxdisp=%-4d missing=%d unexpected=%d" % (md, nmiss, nune))
    except Exception as exc:
        print("  maxdisp=%-4d fail: %s" % (md, str(exc)[:80]))

if net is None:
    print("[fail] no matching maxdisp")
    raise SystemExit(1)

net = net.eval().to(DEV)
print("params %.3fM" % (sum(p.numel() for p in net.parameters()) / 1e6))


def infer(gl, gr):
    t = torch.from_numpy(gl.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1)
    t = ((t - 0.485) / 0.229).to(DEV)
    u = torch.from_numpy(gr.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1)
    u = ((u - 0.485) / 0.229).to(DEV)
    with torch.no_grad():
        o = net(t, u)
    if isinstance(o, (list, tuple)):
        o = o[-1]
    return o.detach().float().cpu().numpy()[0]


print()
print("=" * 88)
print("MSNet3D @ 640x480 (maxdisp=%d)" % maxdisp)
print("=" * 88)
print("%-12s %-9s %-9s %-9s %-9s" % ("scene", "ms", "EPE", "bad1%", "cov%"))
rows, ts = [], []
for name in SCENES:
    sc = load_scene(name)
    if sc is None:
        continue
    L, R, G = sc
    Ls, Rs, gt = resize_pair(L, R, G, 640)
    if Ls.shape[0] != 480:
        Ls = cv2.resize(Ls, (640, 480), interpolation=cv2.INTER_AREA)
        Rs = cv2.resize(Rs, (640, 480), interpolation=cv2.INTER_AREA)
        g2 = cv2.resize(np.nan_to_num(gt, posinf=0.0), (640, 480), interpolation=cv2.INTER_NEAREST)
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
    print("%-12s %-9.1f %-9.3f %-9.2f %-9.1f" % (name, ts[-1], mm["epe"], mm["bad1"], mm["cov"]))
print("-" * 88)
print("%-12s %-9.1f %-9.3f %-9.2f %-9.1f"
      % ("MEAN", float(np.median(ts)),
         float(np.mean([r["epe"] for r in rows])),
         float(np.mean([r["bad1"] for r in rows])),
         float(np.mean([r["cov"] for r in rows]))))