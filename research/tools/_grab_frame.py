# -*- coding: utf-8 -*-
"""抓在线 policy 帧存盘，供离线对照。"""
import sys, time, json
from pathlib import Path
ROOT = Path(r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body")
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image

# 没法直接进后端进程 —— 改用 policy_state 的 nav_frame 只有 shape。
# 替代：从 record_images / 感知线程的录制里拿。或看 navmesh 是否有 dump 帧端点。
print("需要后端支持帧导出 —— 检查 /vision/frame")