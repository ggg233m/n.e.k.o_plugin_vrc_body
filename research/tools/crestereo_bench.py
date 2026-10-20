# -*- coding: utf-8 -*-
"""CREStereo (Intel Arc A770, OpenVINO GPU) vs SGBM —— 速度与光度残差。

    检查：①GPU 学习型立体在本机能不能跑；②比 SGBM 快还是慢；③质量差多少。
    结果：iter2 快但明显更差；iter10 质量追上 SGBM（同一口径），但慢 3 倍。

为什么用**光度和残差**当质量指标（不依赖真值几何）：
    给定视差图 d，把右图按 d 重采样到左图坐标系，|L − warp(R)| 越小说明 d 越自洽。
    ⚠️ 它只衡量「自洽性」，不直接等于深度精度 —— 平滑但错误的解也可能自洽。
    ⚠️ **中位数会饱和（两边都 0.000）**，必须看**均值**。

用法（需要 .tmp/stereo_models/{x2,x10}/ 下的 combined ONNX，未纳入版本控制）：
    python -m research.tools.crestereo_bench 20261010_033237
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.nav_mapping import disparity_range_px, make_sgbm, stereo_disparity  # noqa: E402

MODELS = ROOT / '.tmp' / 'stereo_models'
#: (目录, 模型标签, 输入高, 输入宽)。combined 模型的约定：init 走半分辨率、next 走全分辨率。
ARMS = (('x2', 'iter2', 720, 1280), ('x10', 'iter10', 720, 1280))


def _tensor(gray: np.ndarray, h: int, w: int) -> np.ndarray:
    x = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.stack([x] * 3, axis=0)[None].astype(np.float32)


def _run(compiled, left: np.ndarray, right: np.ndarray, h: int, w: int) -> np.ndarray:
    lh, lw = h // 2, w // 2
    out = compiled([_tensor(left, lh, lw), _tensor(right, lh, lw),
                    _tensor(left, h, w), _tensor(right, h, w)])[compiled.output(0)]
    return np.abs(out[0, 0])


def _residual(left: np.ndarray, right: np.ndarray, disp: np.ndarray) -> np.ndarray:
    h, w = left.shape
    v, u = np.mgrid[0:h, 0:w]
    ur = u - disp
    ok = (disp > 0.5) & (ur >= 1) & (ur < w - 1)
    warped = cv2.remap(right, ur.astype(np.float32), v.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = np.full(disp.shape, np.nan, np.float64)
    out[ok] = np.abs(left.astype(np.float32) - warped.astype(np.float32))[ok]
    return out


def main() -> int:
    import openvino as ov

    name = sys.argv[1] if len(sys.argv) > 1 else '20261010_033237'
    rec = ROOT / 'navmesh_recordings' / name
    if not rec.exists():
        print('[skip] 录制不存在：', name); return 2
    core = ov.Core()
    if 'GPU' not in core.available_devices:
        print('[skip] 没有 GPU 设备；可用：', core.available_devices); return 2

    compiled = {}
    for sub, tag, h, w in ARMS:
        p = MODELS / sub / ('crestereo_combined_%s_%dx%d.onnx' % (tag, h, w))
        if p.exists():
            compiled[tag] = (core.compile_model(core.read_model(str(p)), 'GPU'), h, w)
        else:
            print('[skip] 缺模型：', p)

    w, h = 1440, 810
    print('%-5s %-30s %-9s %-10s %-8s' % ('kf', 'method', 'ms', 'mean_resid', 'valid%'))
    for k in (0, 20, 60):
        left0 = cv2.imread(str(rec / 'kf_img' / ('%06d_L.png' % k)), cv2.IMREAD_GRAYSCALE)
        right0 = cv2.imread(str(rec / 'kf_img' / ('%06d_R.png' % k)), cv2.IMREAD_GRAYSCALE)
        if left0 is None or right0 is None:
            continue
        left = cv2.resize(left0, (w, h), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right0, (w, h), interpolation=cv2.INTER_LINEAR)

        matcher = make_sgbm(disparity_range_px(405.0))
        t0 = time.perf_counter()
        ds = stereo_disparity(left, right, matcher)
        t = (time.perf_counter() - t0) * 1e3
        rs = _residual(left, right, ds)
        print('%-5d %-30s %-9.1f %-10.3f %-8.1f'
              % (k, 'SGBM@1440x810 CPU', t, np.nanmean(rs), 100 * np.isfinite(rs).mean()))

        for tag, (model, mh, mw) in compiled.items():
            t0 = time.perf_counter()
            dc = _run(model, left0, right0, mh, mw)
            t = (time.perf_counter() - t0) * 1e3
            # 视差是「模型输入宽」下的像素数，换到 1440 要按宽度比缩放
            d1440 = cv2.resize(dc, (w, h), interpolation=cv2.INTER_LINEAR) * (w / float(mw))
            rc = _residual(left, right, d1440)
            both = np.isfinite(rs) & np.isfinite(rc)
            print('%-5d %-30s %-9.1f %-10.3f %-8.1f  (共同像素：sgbm %.3f cre %.3f)'
                  % (k, 'CREStereo %s GPU' % tag, t, np.nanmean(rc),
                     100 * np.isfinite(rc).mean(), np.mean(rs[both]), np.mean(rc[both])))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())