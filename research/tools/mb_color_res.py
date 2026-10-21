# -*- coding: utf-8 -*-
"""Middlebury 2014 colour stereo: true RGB vs gray-replicated, across resolutions.

Why this exists
---------------
Every learned-stereo number produced in this repo so far fed the network
``gray`` replicated into 3 channels.  AANet / CREStereo / FFS are trained on
real RGB with ImageNet channel statistics, so replicated grey is an
out-of-distribution input.  Middlebury 2014 ships real colour pairs *and*
dense ground-truth disparity, so we can finally measure:

  1. how much true colour changes each matcher, vs grey replicated to 3ch;
  2. how that changes with resolution;
  3. whether the photometric self-consistency residual used everywhere in
     this repo actually tracks real error (it was never checked against GT).

Ground truth
------------
``disp0.pfm`` is the dense disparity of im0 w.r.t. im1, in pixels at full
resolution, already including ``doffs``.  Verified on Adirondack by warping:
d = +disp0 -> resid 9.36, d = -disp0 -> 32.87, d = disp0 - doffs -> 32.20.
``inf`` marks occluded / out-of-frame pixels and is excluded.

Single-channel input
--------------------
IMREAD_GRAYSCALE vs IMREAD_COLOR changes SGBM's internal kernel selection,
which was worth a reproducible 6.4-7.8% on the VRChat recordings.  This
checks whether that reproduces on an unrelated dataset.

Usage
-----
    .venv_xpu/Scripts/python.exe -u research/tools/mb_color_res.py sgbm
    .venv_xpu/Scripts/python.exe -u research/tools/mb_color_res.py dl
    .venv_xpu/Scripts/python.exe -u research/tools/mb_color_res.py both
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

MB = os.path.join(ROOT, '.tmp', 'mb2014', 'ex')

IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

AANET_SRC = os.path.join(ROOT, '.tmp', 'aanet_src', 'aanet-master')
CKPT_DIR = os.path.join(ROOT, '.tmp', 'stereo_models', 'aanet')

MODELS = (
    ('StereoNetAA', 'stereonet_aa_sceneflow.pth',
     dict(max_disp=192, num_downsample=2, feature_type='stereonet',
          aggregation_type='adaptive', num_scales=1, num_fusions=4,
          deformable_groups=2, mdconv_dilation=2, refinement_type='stereonet',
          no_intermediate_supervision=False, num_stage_blocks=1,
          num_deform_blocks=4, feature_pyramid=False), 1),
    ('AANet', 'aanet_sceneflow.pth',
     dict(max_disp=192, num_downsample=2, feature_type='aanet',
          aggregation_type='adaptive', num_scales=3, num_fusions=6,
          deformable_groups=2, mdconv_dilation=2, refinement_type='stereodrnet',
          no_intermediate_supervision=False, num_stage_blocks=1,
          num_deform_blocks=3, feature_pyramid=False,
          feature_pyramid_network=True), 64),
)

SCENES = ['Adirondack', 'Bicycle1', 'Classroom1', 'Couch', 'Flowers',
          'Motorcycle', 'Piano', 'Playtable', 'Recycle', 'Shelves']

#: Target widths.  Middlebury 2014 is 2880x1988; the VRChat rig is 720x405.
WIDTHS = [2880, 1440, 960, 720, 480, 360]
DL_WIDTHS = [1440, 960, 720, 480]


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def read_pfm(path):
    """PFM -> float32 HxW.  Only the single-channel variant is needed."""
    with open(path, 'rb') as f:
        f.readline()
        w, h = [int(x) for x in f.readline().split()]
        scale = float(f.readline().rstrip())
        raw = f.read(w * h * 4)
    img = np.frombuffer(raw, dtype='<f4').reshape(h, w).astype(np.float32)
    return np.flipud(img).copy()


def load_scene(name):
    d = os.path.join(MB, name + '-perfect')
    lp = os.path.join(d, 'im0.png')
    rp = os.path.join(d, 'im1.png')
    gp = os.path.join(d, 'disp0.pfm')
    if not (os.path.exists(lp) and os.path.exists(rp) and os.path.exists(gp)):
        return None
    L = cv2.imread(lp, cv2.IMREAD_COLOR)
    R = cv2.imread(rp, cv2.IMREAD_COLOR)
    G = read_pfm(gp)
    if L is None or R is None or G is None:
        return None
    return L, R, G


def resize_pair(L, R, G, width):
    """Resize a colour pair and its GT to ``width``.  Disparity scales with width."""
    h0, w0 = G.shape
    if width == w0:
        gt = G.copy()
        Ls, Rs = L, R
    else:
        s = width / float(w0)
        hh = int(round(h0 * s))
        Ls = cv2.resize(L, (width, hh), interpolation=cv2.INTER_AREA)
        Rs = cv2.resize(R, (width, hh), interpolation=cv2.INTER_AREA)
        valid = np.isfinite(G).astype(np.float32)
        Gs = cv2.resize(np.nan_to_num(G, nan=0.0, posinf=0.0), (width, hh),
                        interpolation=cv2.INTER_AREA) * s
        vs = cv2.resize(valid, (width, hh), interpolation=cv2.INTER_AREA)
        gt = np.where(vs > 0.999, Gs, np.inf).astype(np.float32)
    gt = np.where(np.isfinite(gt) & (gt > 0), gt, np.inf).astype(np.float32)
    return Ls, Rs, gt


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def gt_metrics(est, gt, min_disp=1.0):
    """Errors against ground-truth disparity.  ``est <= min_disp`` means invalid."""
    ok_gt = np.isfinite(gt)
    ok_est = est > min_disp
    both = ok_gt & ok_est
    n_gt = int(ok_gt.sum())
    cov = 100.0 * both.sum() / max(n_gt, 1)
    if both.sum() == 0:
        return dict(cov=cov, epe=float('nan'), bad1=float('nan'),
                    bad3=float('nan'), rmse=float('nan'), n=0)
    e = np.abs(est[both].astype(np.float64) - gt[both].astype(np.float64))
    return dict(cov=cov, epe=float(e.mean()),
                bad1=100.0 * float((e > 1.0).mean()),
                bad3=100.0 * float((e > 3.0).mean()),
                rmse=float(np.sqrt((e * e).mean())), n=int(both.sum()))


def photo_resid(L, R, disp):
    """Photometric self-consistency residual: the metric used before GT existed."""
    h, w = disp.shape
    v, u = np.mgrid[0:h, 0:w]
    ur = u - disp
    ok = (disp > 1.0) & (ur >= 1) & (ur < w - 1)
    wr = cv2.remap(R, ur.astype(np.float32), v.astype(np.float32),
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if L.ndim == 3:
        diff = np.abs(L.astype(np.float32) - wr.astype(np.float32)).mean(axis=2)
    else:
        diff = np.abs(L.astype(np.float32) - wr.astype(np.float32))
    if not ok.any():
        return float('nan')
    return float(diff[ok].mean())


# --------------------------------------------------------------------------
# SGBM
# --------------------------------------------------------------------------
def sgbm_for(gt):
    """numDisparities sized from the GT range, so no method is disparity-starved."""
    v = gt[np.isfinite(gt)]
    hi = float(np.percentile(v, 99.9)) if v.size else 64.0
    nd = int(np.ceil(max(hi * 1.10, 32) / 16.0) * 16)
    nd = min(max(nd, 32), 384)
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=5,
        P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY), nd


def run_sgbm(scenes, widths):
    print()
    print('=' * 116)
    print('SGBM on Middlebury 2014 -- 1ch(gray) vs 3ch(colour) vs 3ch(gray replicated)')
    print('metric vs dense GT; cov = pct of GT-valid px where an estimate exists')
    print('=' * 116)
    hdr = ('%-11s %-6s %-8s %-7s %-7s %-8s %-7s %-7s %-6s'
           % ('scene', 'W', 'input', 'ms', 'cov%', 'EPE', 'bad1%', 'bad3%', 'nd'))
    print(hdr)
    print('-' * len(hdr))
    agg = {}
    for name in scenes:
        sc = load_scene(name)
        if sc is None:
            print('%-11s MISSING' % name)
            continue
        L, R, G = sc
        for w in widths:
            Ls, Rs, gt = resize_pair(L, R, G, w)
            m, nd = sgbm_for(gt)
            lg = cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY)
            rg = cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY)
            lg3 = cv2.cvtColor(lg, cv2.COLOR_GRAY2BGR)
            rg3 = cv2.cvtColor(rg, cv2.COLOR_GRAY2BGR)
            for tag, a, b in (('1ch', lg, rg), ('3ch', Ls, Rs), ('gray3', lg3, rg3)):
                ts = []
                for _i in range(3):
                    t0 = time.perf_counter()
                    d = m.compute(a, b).astype(np.float32) / 16.0
                    ts.append((time.perf_counter() - t0) * 1e3)
                mm = gt_metrics(d, gt)
                agg.setdefault((w, tag), []).append(mm)
                print('%-11s %-6d %-8s %-7.1f %-7.1f %-8.3f %-7.2f %-7.2f %-6d'
                      % (name, w, tag, float(np.median(ts)), mm['cov'],
                         mm['epe'], mm['bad1'], mm['bad3'], nd))
    print()
    print('---- aggregate, mean over scenes ----')
    hdr2 = '%-7s %-9s %-8s %-8s %-8s' % ('W', 'input', 'cov%', 'EPE', 'bad1%')
    print(hdr2)
    print('-' * len(hdr2))
    for w in widths:
        base = None
        for tag in ('1ch', '3ch', 'gray3'):
            rows = agg.get((w, tag))
            if not rows:
                continue
            cov = float(np.mean([r['cov'] for r in rows]))
            epe = float(np.mean([r['epe'] for r in rows]))
            b1 = float(np.mean([r['bad1'] for r in rows]))
            if tag == '1ch':
                base = epe
            note = '' if base is None else '   EPE %+6.1f%% vs 1ch' % (
                100.0 * (epe - base) / base)
            print('%-7d %-9s %-8.1f %-8.3f %-8.2f%s' % (w, tag, cov, epe, b1, note))
        print()
    return agg


# --------------------------------------------------------------------------
# learned models
# --------------------------------------------------------------------------
def build_torch(torch, ckpt, cfg):
    sys.path.insert(0, AANET_SRC)
    from nets.aanet import AANet  # noqa: E402
    model = AANet(**cfg)
    state = torch.load(os.path.join(CKPT_DIR, ckpt), map_location='cpu',
                       weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.eval().to('xpu')


def to_tensor(torch, bgr, align, half=False):
    """BGR uint8 -> normalised NCHW.  Edge-padded to a multiple of ``align``."""
    h, w = bgr.shape[:2]
    hh = (h + align - 1) // align * align
    ww = (w + align - 1) // align * align
    if (hh, ww) != (h, w):
        bgr = cv2.copyMakeBorder(bgr, 0, hh - h, 0, ww - w, cv2.BORDER_REPLICATE)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1)[None]
    t = (t - torch.from_numpy(IM_MEAN)) / torch.from_numpy(IM_STD)
    return (t.half() if half else t).to('xpu')


def as_gray3(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def run_dl(scenes, widths, half=False):
    import torch
    if not torch.xpu.is_available():
        print('[skip] torch XPU unavailable')
        return
    if not os.path.isdir(AANET_SRC):
        print('[skip] missing AANet source')
        return
    align_of = dict((m[0], m[3]) for m in MODELS)
    models = {}
    for nm, ck, cfg, _al in MODELS:
        try:
            models[nm] = build_torch(torch, ck, cfg)
        except Exception as exc:
            print('[skip] %s: %s' % (nm, str(exc)[:90]))
    if not models:
        return
    print()
    print('=' * 122)
    print('Learned stereo on Middlebury 2014 -- true RGB vs gray replicated to 3ch')
    print('photo = photometric self-consistency residual (NOT accuracy, see report sec.6)')
    print('=' * 122)
    hdr = ('%-11s %-6s %-13s %-7s %-7s %-7s %-8s %-7s %-7s %-7s'
           % ('scene', 'W', 'model', 'input', 'ms', 'cov%', 'EPE', 'bad1%', 'bad3%', 'photo'))
    print(hdr)
    print('-' * len(hdr))
    agg = {}
    for name in scenes:
        sc = load_scene(name)
        if sc is None:
            continue
        L, R, G = sc
        for w in widths:
            Ls, Rs, gt = resize_pair(L, R, G, w)
            for nm in sorted(models):
                model = models[nm]
                align = align_of[nm]
                for tag, A, B in (('rgb', Ls, Rs), ('gray3', as_gray3(Ls), as_gray3(Rs))):
                    x1 = to_tensor(torch, A, align)
                    x2 = to_tensor(torch, B, align)
                    with torch.no_grad():
                        model(x1, x2)
                    torch.xpu.synchronize()
                    ts = []
                    for _i in range(3):
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            out = model(x1, x2)
                        torch.xpu.synchronize()
                        ts.append((time.perf_counter() - t0) * 1e3)
                    disp = out[-1].detach().float().cpu().numpy()[0]
                    disp = disp[:gt.shape[0], :gt.shape[1]]
                    mm = gt_metrics(disp, gt)
                    ph = photo_resid(cv2.cvtColor(Ls, cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(Rs, cv2.COLOR_BGR2GRAY),
                                     np.nan_to_num(disp, nan=0.0, posinf=0.0))
                    agg.setdefault((nm, w, tag), []).append(mm)
                    print('%-11s %-6d %-13s %-7s %-7.1f %-7.1f %-8.3f %-7.2f %-7.2f %-7.3f'
                          % (name, w, nm, tag, float(np.median(ts)), mm['cov'],
                             mm['epe'], mm['bad1'], mm['bad3'], ph))
    print()
    print('---- aggregate, mean over scenes ----')
    hdr2 = '%-13s %-7s %-9s %-8s %-8s %-8s' % ('model', 'W', 'input', 'cov%', 'EPE', 'bad1%')
    print(hdr2)
    print('-' * len(hdr2))
    for nm in sorted(models):
        for w in widths:
            base = None
            for tag in ('gray3', 'rgb'):
                rows = agg.get((nm, w, tag))
                if not rows:
                    continue
                cov = float(np.mean([r['cov'] for r in rows]))
                epe = float(np.mean([r['epe'] for r in rows]))
                b1 = float(np.mean([r['bad1'] for r in rows]))
                if tag == 'gray3':
                    base = epe
                note = '' if base is None else '   EPE %+6.1f%% vs gray3' % (
                    100.0 * (epe - base) / base)
                print('%-13s %-7d %-9s %-8.1f %-8.3f %-8.2f%s'
                      % (nm, w, tag, cov, epe, b1, note))
            print()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if mode in ('sgbm', 'both'):
        run_sgbm(SCENES, WIDTHS)
    if mode in ('dl', 'both'):
        run_dl(SCENES, DL_WIDTHS)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
