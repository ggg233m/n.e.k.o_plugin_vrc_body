# -*- coding: utf-8 -*-
"""LAS2-M / H across resolutions: latency scaling and accuracy.

Rig-native 16:9 long sides.  max_disp must grow with resolution or the near
field gets clipped, so scale it with fx exactly like nav_mapping does."""
import os, sys, time, numpy as np, cv2, torch
import torch.nn.functional as F
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.abspath(".")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, ".tmp", "las_src", "LiteAnyStereo"))

from core.models import build_model
from research.tools.mb_color_res import load_scene, resize_pair, gt_metrics, SCENES


class Pad:
    def __init__(self, dims, divis_by=32):
        self.ht, self.wd = dims[-2:]
        ph = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pw = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]

    def pad(self, *xs):
        return [F.pad(x, self._pad, mode="replicate") for x in xs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


DEV = torch.device("xpu")
CKPT = os.path.join(ROOT, "las_ckpt")


def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.as_tensor(rgb, device=DEV).float()[None].permute(0, 3, 1, 2)


def bench(size, maxdisp, W, H):
    model = build_model("las2", model_size=size, max_disp=maxdisp)
    sd = torch.load(os.path.join(CKPT, "LAS2_%s.pth" % size.upper()),
                    map_location="cpu", weights_only=False)
    if "model" in sd:
        sd = sd["model"]
    if sd and all(k.startswith("module.") for k in sd):
        sd = dict((k[7:], v) for k, v in sd.items())
    model.load_state_dict(sd, strict=True)
    model = model.to(DEV).eval()

    rows, ts = [], []
    for name in SCENES:
        sc = load_scene(name)
        if sc is None:
            continue
        L, R, G = sc
        Ls, Rs, gt = resize_pair(L, R, G, W)
        if Ls.shape[0] != H:
            Ls = cv2.resize(Ls, (W, H), interpolation=cv2.INTER_AREA)
            Rs = cv2.resize(Rs, (W, H), interpolation=cv2.INTER_AREA)
            g2 = cv2.resize(np.nan_to_num(gt, posinf=0.0), (W, H), interpolation=cv2.INTER_NEAREST)
            gt = np.where(g2 > 0, g2, np.inf).astype(np.float32)
        x1, x2 = to_tensor(Ls), to_tensor(Rs)
        padder = Pad(x1.shape, 32)
        x1p, x2p = padder.pad(x1, x2)
        with torch.no_grad():
            model(x1p, x2p)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x1p, x2p)
        torch.xpu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
        d = out[-1] if isinstance(out, (list, tuple)) else out
        d = padder.unpad(d).detach().float().cpu().numpy()[0, 0]
        mm = gt_metrics(d, gt)
        ok = np.isfinite(gt) & (d > 1.0)
        mm["relZ"] = (float(np.median(np.abs(d[ok] / gt[ok] - 1.0)) * 100)
                      if ok.sum() > 500 else float("nan"))
        mm["scene"] = name
        rows.append(mm)
    return rows, float(np.median(ts))


LADDER = [(640, 480, 192), (1080, 608, 192), (1440, 810, 256)]

for size in ("m", "h"):
    print()
    print("=" * 100)
    print("LAS2-%s across rig long sides" % size.upper())
    print("=" * 100)
    print("%-12s %-10s %-9s %-9s %-9s %-9s %-10s" %
          ("long side", "max_disp", "ms", "FPS", "EPE", "bad1%", "relZ%"))
    print("-" * 100)
    for W, H, md in LADDER:
        try:
            rows, ms = bench(size, md, W, H)
        except Exception as exc:
            print("%-12d %-10d FAILED %s" % (W, md, str(exc)[:60]))
            continue
        print("%-12d %-10d %-9.1f %-9.2f %-9.3f %-9.2f %-10.2f"
              % (W, md, ms, 1000.0 / ms,
                 float(np.mean([r["epe"] for r in rows])),
                 float(np.mean([r["bad1"] for r in rows])),
                 float(np.nanmean([r["relZ"] for r in rows]))))