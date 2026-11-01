# -*- coding: utf-8 -*-
"""对照实验：同一模型，OpenVR mirror 当前帧 vs 离线噪声帧，看模型输出差多少。"""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body")
sys.path.insert(0, str(ROOT / "research" / "tools"))
import torch
from policy_runtime import PolicyRuntime

CK  = r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3"
RUN = r"G:\vrc_recording\vrc_model\runs\m0_s3_448\best"

print("[1] loading runtime (graph=0, 同在线配置)", flush=True)
rt = PolicyRuntime(CK, RUN, compile=True, merge=True, graph=False)

import openvr
sys.path.insert(0, str(ROOT))
from backend.openvr_mirror import MirrorEye, create_device, read_stereo
openvr.init(openvr.VRApplication_Background)
device, context = create_device()
comp = openvr.VRCompositor()
eyes = [MirrorEye(comp, device, context, e, gpu_downscale=(1080, 1))
        for e in (openvr.Eye_Left, openvr.Eye_Right)]
print("[2] mirror open", flush=True)

def grab():
    pair = read_stereo(eyes[0], eyes[1])
    return None if pair is None else pair[0]

# 真实当前帧
cur = grab()
print("[3] frame:", None if cur is None else cur.shape, flush=True)
if cur is not None:
    Image = __import__("PIL.Image", fromlist=["Image"])
    __import__("PIL.Image").Image.fromarray(cur).save(str(ROOT / ".tmp" / "online_frame.jpg"), quality=90)
    # A: 真实帧 + 真实量级 wp
    outA = rt.act(cur, cur, (0.22, 0.78), (0.0, 0.0))
    print("A real-frame  wp=(0.22,0.78): step0=%s lat=%.0f" % (outA["block"][0], outA["latency_ms"]), flush=True)
    # B: 真实帧 + dry-run 同款 wp
    outB = rt.act(cur, cur, (0.0, 0.8), (0.0, 0.0))
    print("B real-frame  wp=(0.00,0.80): step0=%s" % (outB["block"][0],), flush=True)

# C: 噪声帧对照（dry-run 首测时的量级）
rng = np.random.default_rng(0)
noise = rng.integers(0, 255, (606, 1080, 3), dtype=np.uint8)
outC = rt.act(noise, noise, (0.22, 0.78), (0.0, 0.0))
print("C noise-frame wp=(0.22,0.78): step0=%s" % (outC["block"][0],), flush=True)
for m in eyes: m.close()
openvr.shutdown()
print("DONE", flush=True)