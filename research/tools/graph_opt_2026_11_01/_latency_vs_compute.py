# -*- coding: utf-8 -*-
"""关键判别：LM 图是「算不动」还是「在等」。

三个独立证据：
  1. 单次 replay vs 连续 N 次 replay（不 sync）：若单次 ~67ms 但连发 N 次平均远低于它，
     说明 GPU 空闲着等 CPU —— 延迟受限。
  2. 两个 CUDA/XPU 流并发 replay：若总时间 ≈ 单次，说明同一流内序列化。
  3. 逐层拆解：24 层里哪些是真耗时。
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body")
sys.path.insert(0, str(ROOT / "research" / "tools"))
from policy_runtime import setup_oneapi
setup_oneapi(graph=True)

import torch
from transformers import AutoProcessor
from train_policy_lora import load_model

CK  = r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3"
RUN = r"G:\vrc_recording\vrc_model\runs\m0_s3_448\best"
DEV = "xpu"; MAXLEN=192

model, head, _ = load_model(CK, 128, DEV)
model.load_adapter(RUN + r"\adapter", adapter_name="trained", is_trainable=False)
model.set_adapter("trained")
model = model.merge_and_unload()
model.eval(); head.eval()
inner = model.base_model
lm = inner.language_model

res = {}
E  = torch.randn((1, MAXLEN, 1024), dtype=torch.bfloat16, device=DEV)
AM = torch.ones((1, MAXLEN), dtype=torch.long, device=DEV)
P  = torch.zeros((3, 1, MAXLEN), dtype=torch.long, device=DEV)

stream = torch.xpu.Stream()
stream.wait_stream(torch.xpu.current_stream())

# 建 LM 图
with torch.inference_mode(), torch.xpu.stream(stream):
    for _ in range(3):
        lm(inputs_embeds=E, attention_mask=AM, position_ids=P, use_cache=False)
stream.synchronize()
g = torch.xpu.XPUGraph()
with torch.inference_mode(), torch.xpu.graph(g, stream=stream):
    out = lm(inputs_embeds=E, attention_mask=AM, position_ids=P, use_cache=False)
torch.xpu.synchronize(); g.replay(); torch.xpu.synchronize()

# ---- 证据 1：单次（带 sync） vs 连发 20 次（只在末尾 sync）----
def once():
    torch.xpu.synchronize(); t = time.perf_counter()
    g.replay(); torch.xpu.synchronize()
    return (time.perf_counter() - t) * 1e3

ts = [once() for _ in range(30)]
ts.sort()
res["single_sync_ms"] = {"p50": round(ts[len(ts)//2], 2), "min": round(ts[0], 2)}

N = 20
torch.xpu.synchronize(); t = time.perf_counter()
for _ in range(N):
    g.replay()
torch.xpu.synchronize()
batch = (time.perf_counter() - t) * 1e3
res["batched_20_ms"] = {"total": round(batch, 2), "per_call": round(batch/N, 2)}
res["async_ratio"] = round(res["single_sync_ms"]["p50"] / (batch/N), 2)
print("[1] single=%.1fms  batched_per=%.1fms  ratio=%.2fx" % (
    res["single_sync_ms"]["p50"], batch/N, res["async_ratio"]), flush=True)

# ---- 证据 2：CPU 提交耗时（完全不 sync）----
N = 100
t0 = time.perf_counter()
for _ in range(N):
    g.replay()
cpu_us = (time.perf_counter() - t0) * 1e6 / N
torch.xpu.synchronize()
res["cpu_submit_us"] = round(cpu_us, 1)
print("[2] CPU submit = %.1f us/call" % cpu_us, flush=True)

# ---- 证据 3：两流并发 ----
def two_stream_test():
    s2 = torch.xpu.Stream()
    s2.wait_stream(torch.xpu.current_stream())
    torch.xpu.synchronize()
    t = time.perf_counter()
    g.replay()
    with torch.xpu.stream(s2):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) * 1e3

tt = [two_stream_test() for _ in range(10)]
tt.sort()
res["two_stream_ms"] = round(tt[len(tt)//2], 2)
res["two_vs_one"] = round(res["two_stream_ms"] / res["single_sync_ms"]["p50"], 2)
print("[3] two-stream = %.1fms (vs single %.1fms, ratio %.2f)" % (
    res["two_stream_ms"], res["single_sync_ms"]["p50"], res["two_vs_one"]), flush=True)

print("RESULT " + json.dumps(res, ensure_ascii=False), flush=True)