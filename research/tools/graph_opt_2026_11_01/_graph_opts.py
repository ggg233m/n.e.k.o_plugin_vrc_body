# -*- coding: utf-8 -*-
"""测 torch.compile 的三种"图优化"配置，看 CPU 派发能降多少。
   A. default
   B. cpp_wrapper=True      （C++ wrapper 代替 Python wrapper，直接砍 Python 派发税）
   C. aggressive_fusion + max_fusion_size=256（多融 kernel，减少 kernel 数）
   D. reduce-overhead （看它是不是静默回退）
"""
import torch, time, sys, os, json, numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
sys.path.insert(0, r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\research\tools")
from policy_runtime import to_policy_rgb, PROMPT

TAG = sys.argv[1]
CK  = r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3"
RUN = r"G:\vrc_recording\vrc_model\runs\m0_s3_448\best"
dev = torch.device("xpu"); DT = torch.bfloat16
SIZE=(448,256); MAXLEN=192

proc = AutoProcessor.from_pretrained(CK)
pad_id = int(getattr(proc.tokenizer,"pad_token_id",0) or 0)
rng = np.random.default_rng(0)
cur  = to_policy_rgb(rng.integers(0,255,(606,1077,3),dtype=np.uint8), SIZE)
prev = to_policy_rgb(rng.integers(0,255,(606,1077,3),dtype=np.uint8), SIZE)
tmpl = proc.apply_chat_template([{"role":"user","content":[{"type":"video"},{"type":"text","text":PROMPT%(0.,0.,0.3,0.5)}]}],
                                tokenize=False, add_generation_prompt=True)
mp=SIZE[0]*SIZE[1]
inp = proc(text=[tmpl], videos=[np.stack([prev,cur])], return_tensors="pt", padding=True,
           min_pixels=mp, max_pixels=mp, cap_pixels_per_frame=False, do_sample_frames=False)
ids = inp["input_ids"][0].tolist(); n=len(ids)
ids=ids+[pad_id]*(MAXLEN-n); mask=[1]*n+[0]*(MAXLEN-n)
kw = dict(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([mask]),
          pixel_values_videos=inp["pixel_values_videos"], video_grid_thw=inp["video_grid_thw"])
if "mm_token_type_ids" in inp:
    kw["mm_token_type_ids"]=torch.tensor([inp["mm_token_type_ids"][0].tolist()+[0]*(MAXLEN-n)])
kw={k:v.to(dev) for k,v in kw.items()}

base = AutoModelForImageTextToText.from_pretrained(CK, dtype=DT).to(dev).eval()
p = PeftModel.from_pretrained(base, RUN+r"\adapter", adapter_name="trained")
p.set_adapter("trained"); m = p.merge_and_unload(); m.eval()
inner = m.model

import torch._inductor.config as ic
opts = {}
if TAG == "B_cppwrapper":
    ic.cpp_wrapper = True
    opts["cpp_wrapper"] = True
elif TAG == "C_aggrfuse":
    ic.aggressive_fusion = True
    ic.max_fusion_size = 256
    opts["aggressive_fusion"] = True; opts["max_fusion_size"] = 256
elif TAG == "D_reduceoverhead":
    opts["mode"] = "reduce-overhead"

try:
    compiled = torch.compile(inner, **opts) if opts.get("mode") else torch.compile(inner)
except Exception as e:
    print("COMPILE_SETUP_FAIL", type(e).__name__, str(e)[:150], flush=True); raise SystemExit(1)

def cpu_only(fn, iters=20, warm=6):
    for _ in range(warm): fn()
    torch.xpu.synchronize()
    ts=[]
    for _ in range(iters):
        t=time.perf_counter(); fn(); ts.append((time.perf_counter()-t)*1e3)
    torch.xpu.synchronize(); ts.sort(); return ts[len(ts)//2]

def wall(fn, iters=20, warm=6):
    ts=[]
    for i in range(warm+iters):
        torch.xpu.synchronize(); t=time.perf_counter(); fn(); torch.xpu.synchronize()
        if i>=warm: ts.append(time.perf_counter()-t)
    ts.sort(); return ts[len(ts)//2]*1e3

def kernels(fn):
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        fn(); torch.xpu.synchronize()
    ka = prof.key_averages()
    return sum(e.count for e in ka)

f = lambda: compiled(**kw, use_cache=False).last_hidden_state
try:
    with torch.inference_mode():
        t0=time.perf_counter(); f(); torch.xpu.synchronize()
        compile_s = time.perf_counter()-t0
        c = cpu_only(f); w = wall(f); k = kernels(f)
    print("RESULT %s compile=%.0fs cpu=%.1f wall=%.1f kernel=%d" % (TAG, compile_s, c, w, k), flush=True)
except Exception as e:
    print("RUN_FAIL %s %s %s" % (TAG, type(e).__name__, str(e)[:200].replace(chr(10)," ")), flush=True)
