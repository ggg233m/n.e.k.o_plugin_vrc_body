# -*- coding: utf-8 -*-
"""Lightweight stereo on Intel Arc via native PyTorch XPU (AANet family).

    先决条件（都不在版本控制里）：
      .venv_xpu/                        torch 2.14.1+xpu（python 3.11）
      .tmp/aanet_src/aanet-master/      https://github.com/haofeixu/aanet
      .tmp/stereo_models/aanet/*.pth    MODEL_ZOO.md 里的 Google Drive 链接

    ⚠️ 上游 nets/deform_conv 是 CUDA 扩展。本仓库把它换成了纯 PyTorch 实现
       （.tmp/aanet_src/aanet-master/nets/deform_conv/deform_conv.py，备份见 *.cuda.bak）。
       等价性自检：零 offset + mask=1 时与 conv2d 差 7.2e-07。

    用法：
        .venv_xpu/Scripts/python.exe -m research.tools.aanet_bench 20261010_033237
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

AANET_SRC = ROOT / '.tmp' / 'aanet_src' / 'aanet-master'
CKPT = ROOT / '.tmp' / 'stereo_models' / 'aanet' / 'stereonet_aa_sceneflow.pth'

# inference.py 里的官方归一化
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

#: StereoNet-AA（0.526 M 参数）：从 checkpoint 的形状反推出来的配置。
#: num_scales=1（只有 scale-0 分支）、max_disp=192（-> 48 通道）、num_fusions=4、
#: num_deform_blocks=4、refinement_type='stereonet'。strict=True 加载，missing=0/unexpected=0。
#: 每个 entry = (显示名, 权重文件, AANet 构造参数, 输入尺寸必须整除的数)。
#: StereoNet-AA 与 AANet 本体的结构**不一样**，都是从那本 checkpoint 的张量形状反推的。
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


def build(ckpt: str, cfg: dict, half: bool = False) -> torch.nn.Module:
    sys.path.insert(0, str(AANET_SRC))
    from nets.aanet import AANet  # noqa: E402

    model = AANet(**cfg)
    state = torch.load(str(ROOT / '.tmp' / 'stereo_models' / 'aanet' / ckpt),
                       map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval().to('xpu')
    if half:
        model.half()
    return model


def prep(gray: np.ndarray, half: bool = False) -> torch.Tensor:
    t = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1)
    t = (t - IM_MEAN) / IM_STD
    return (t.half() if half else t).to('xpu')


def residual(left: np.ndarray, right: np.ndarray, disp: np.ndarray):
    """光度自洽残差：按 disp 把右图 warp 到左图，取平均绝对差。"""
    h, w = left.shape
    v, u = np.mgrid[0:h, 0:w]
    ur = u - disp
    ok = (disp > 0.5) & (ur >= 1) & (ur < w - 1)
    warped = cv2.remap(right, ur.astype(np.float32), v.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = np.full(disp.shape, np.nan)
    out[ok] = np.abs(left.astype(np.float32) - warped.astype(np.float32))[ok]
    return float(np.nanmean(out)), float(100 * np.isfinite(out).mean())


def main() -> int:
    from backend.nav_mapping import disparity_range_px, make_sgbm, stereo_disparity

    if not torch.xpu.is_available():
        print('[skip] torch XPU 不可用'); return 2
    if not AANET_SRC.exists():
        print('[skip] 缺 AANet 源码，见模块 docstring'); return 2

    name = sys.argv[1] if len(sys.argv) > 1 else '20261010_033237'
    rec = ROOT / 'navmesh_recordings' / name
    if not rec.exists():
        print('[skip] 录制不存在：', name); return 2

    print('%-5s %-12s %-9s %-9s %-8s %-8s'
          % ('kf', 'method', 'ms', 'resid', 'cov%', 'disp_p50'))
    models = {}
    for k in (0, 20, 60, 120):
        left = cv2.imread(str(rec / 'kf_img' / ('%06d_L.png' % k)), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(rec / 'kf_img' / ('%06d_R.png' % k)), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            continue
        t0 = time.perf_counter()
        ds = stereo_disparity(left, right, make_sgbm(disparity_range_px(202.5)))
        t = (time.perf_counter() - t0) * 1e3
        r = residual(left, right, ds)
        print('%-5d %-12s %-9.1f %-9.3f %-8.1f %-8.2f'
              % (k, 'SGBM', t, r[0], r[1], float(np.median(ds[ds > 1]))))
        for name, ckpt, cfg, align in MODELS:
            if name not in models:
                try:
                    models[name] = build(ckpt, cfg)
                except Exception as exc:
                    models[name] = None
                    print('%-5d %-12s SKIP: %s' % (k, name, str(exc)[:70]))
                    continue
            m = models[name]
            if m is None:
                continue
            l2, r2 = left, right
            if align > 1:
                hh = (left.shape[0] + align - 1) // align * align
                ww = (left.shape[1] + align - 1) // align * align
                if (hh, ww) != left.shape:
                    l2 = cv2.resize(left, (ww, hh), interpolation=cv2.INTER_LINEAR)
                    r2 = cv2.resize(right, (ww, hh), interpolation=cv2.INTER_LINEAR)
            x1, x2 = prep(l2), prep(r2)
            with torch.no_grad():
                m(x1, x2)                      # warmup
            torch.xpu.synchronize()
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = m(x1, x2)
                torch.xpu.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            disp = out[-1].detach().float().cpu()[0].numpy()
            rr = residual(l2, r2, disp)
            print('%-5d %-12s %-9.1f %-9.3f %-8.1f %-8.2f'
                  % (k, name, float(np.median(ts)),
                     rr[0], rr[1], float(np.median(disp))))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())