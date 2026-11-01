# -*- coding: utf-8 -*-
"""确认修复方向：body 用期望速度（wp 方向）能否唤醒模型。"""
import sys, time
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
f1 = grab(); time.sleep(0.4); f2 = grab()
wp = (0.22, 0.78)
print("A body=(0,0)         :", rt.act(f2, f1, wp, (0.0, 0.0))["block"][0], flush=True)
print("B body=(0,-0.5) 期望 :", rt.act(f2, f1, wp, (0.0, -0.5))["block"][0], flush=True)
print("C body=(0,-0.8) 期望 :", rt.act(f2, f1, wp, (0.0, -0.8))["block"][0], flush=True)
print("D body=(0,-1.0) 期望 :", rt.act(f2, f1, wp, (0.0, -1.0))["block"][0], flush=True)
for m in eyes: m.close()
openvr.shutdown()
print("DONE", flush=True)