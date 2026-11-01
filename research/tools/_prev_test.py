# -*- coding: utf-8 -*-
"""验证 prev 帧假设：在线用 0.4s 前真帧，这里模拟两帧有真实位移差的情况。"""
import sys, time, json
from pathlib import Path
import numpy as np
ROOT = Path(r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body")
sys.path.insert(0, str(ROOT / "research" / "tools"))
import torch
from policy_runtime import PolicyRuntime

CK  = r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3"
RUN = r"G:\vrc_recording\vrc_model\runs\m0_s3_448\best"

rt = PolicyRuntime(CK, RUN, compile=True, merge=True, graph=False)

import openvr
sys.path.insert(0, str(ROOT))
from backend.openvr_mirror import MirrorEye, create_device, read_stereo
openvr.init(openvr.VRApplication_Background)
device, context = create_device()
comp = openvr.VRCompositor()
eyes = [MirrorEye(comp, device, context, e, gpu_downscale=(1080, 1))
        for e in (openvr.Eye_Left, openvr.Eye_Right)]

def grab():
    pair = read_stereo(eyes[0], eyes[1])
    return None if pair is None else pair[0]

# 采一对相隔 0.4s 的帧（角色静止，两帧几乎相同 —— 与在线一致）
f1 = grab()
time.sleep(0.4)
f2 = grab()
print("[1] got frame pair", f1.shape, flush=True)

# 在线的真实调用是 act(cur, prev, wp, body) 且 body 来自 OSC（可能非零）
# 逐项复现在线参数：prev=f1(0.4s前), cur=f2, wp=(0.22,0.78), body=(0,0)
out = rt.act(f2, f1, (0.22, 0.78), (0.0, 0.0))
print("A prev=0.4s帧 body=(0,0): step0=%s" % (out["block"][0],), flush=True)

# body 非零（OSC 回传行走中）
out2 = rt.act(f2, f1, (0.22, 0.78), (0.0, -0.73))
print("B prev=0.4s帧 body=(0,-0.73): step0=%s" % (out2["block"][0],), flush=True)

# prev=cur 对照
out3 = rt.act(f2, f2, (0.22, 0.78), (0.0, 0.0))
print("C prev=cur        body=(0,0): step0=%s" % (out3["block"][0],), flush=True)
for m in eyes: m.close()
openvr.shutdown()
print("DONE", flush=True)