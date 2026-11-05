# -*- coding: utf-8 -*-
"""策略图像预处理（**只依赖 numpy + PIL，不 import torch**）。

为什么要单独一个模块（2026-11-05）：
  `to_policy_rgb` 原本在 `policy_runtime.py` 里，而后者顶部就 `import torch`。
  `.venv` 主后端没有 torch，但它需要在**把帧发给 worker 之前**先缩到 448×256 ——
  否则每拍要经管道传 WGC 原图（1920×1080 时 base64 约 8 MB，2 Hz 下编码本身
  就要几十毫秒，白吃延迟）。

  复制一份实现是绝对不行的：这个函数的几何（中心裁 16:9 → 缩到内容高 → 对称补黑边）
  必须与训练侧逐像素一致，两份实现必然静默漂移（本仓已因同类问题踩过坑）。
  所以把**唯一那份**搬到这里，`policy_runtime.py` 改为从这里 import。
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def to_policy_rgb(rgb: np.ndarray, size=(448, 256)) -> np.ndarray:
    """同一几何：中心裁 16:9（1077x606）→ 缩到 Wx内容高 → 对称补黑边到 WxH。

    XPU venv 没有 cv2；PIL 的 resize 在下采样时按缩放系数放大滤波核（抗锯齿），
    与训练侧 ffmpeg 的 area 缩放同属面积平均族，几何完全一致。
    """
    w, h = int(size[0]), int(size[1])
    content_h = min(h, int(round(w * 9 / 16)))
    content_h -= content_h % 2
    top = (h - content_h) // 2
    src_h, src_w = rgb.shape[:2]
    crop_w = min(int(round(src_h * 16 / 9)), src_w)
    left = (src_w - crop_w) // 2
    crop = rgb[:, left:left + crop_w]
    im = Image.fromarray(crop).resize((w, content_h), Image.BILINEAR)
    out = np.zeros((h, w, 3), np.uint8)
    out[top:top + content_h] = np.asarray(im)
    return out
