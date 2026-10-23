# -*- coding: utf-8 -*-
"""LAS2 with official preprocessing, InputPadder inlined (avoids scipy dep)."""
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


def bench(size, h, w):
    model = build_model("las2", model_size=size, max_disp=192)
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
        Ls, Rs, gt = resize_pair(L, R, G, w)
        if Ls.shape[0] != h:
            Ls = cv2.resize(Ls, (w, h), interpolation=cv2.INTER_AREA)
            Rs = cv2.resize(Rs, (w, h), interpolation=cv2.INTER_AREA)
            g2 = cv2.resize(np.nan_to_num(gt, posinf=0.0), (w, h), interpolation=cv2.INTER_NEAREST)
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
        mm["scene"] = name
        rows.append(mm)
    return rows, float(np.median(ts))


for size in ("s", "m", "h"):
    print()
    print("=" * 86)
    print("LAS2-%s @ 640x480  (official preprocessing: 0..255, no mean/std)" % size.upper())
    print("=" * 86)
    try:
        rows, ms = bench(size, 480, 640)
    except Exception as exc:
        print("  FAILED:", type(exc).__name__, str(exc)[:200])
        continue
    print("%-12s %-9s %-9s %-9s" % ("scene", "EPE", "bad1%", "cov%"))
    for r in rows:
        print("%-12s %-9.3f %-9.2f %-9.1f" % (r["scene"], r["epe"], r["bad1"], r["cov"]))
    print("-" * 86)
    print("%-12s %-9.3f %-9.2f %-9.1f   (ms %.1f)"
          % ("MEAN", float(np.mean([r["epe"] for r in rows])),
             float(np.mean([r["bad1"] for r in rows])),
             float(np.mean([r["cov"] for r in rows])), ms))