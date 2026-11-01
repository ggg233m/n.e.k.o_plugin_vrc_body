# -*- coding: utf-8 -*-
"""端到端验证：PolicyRuntime(graph=True) 双图 vs eager 正确性 + 延迟。"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"H:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body")
sys.path.insert(0, str(ROOT / "research" / "tools"))

from policy_runtime import PolicyRuntime

CK  = r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3"
RUN = r"G:\vrc_recording\vrc_model\runs\m0_s3_448\best"

rng = np.random.default_rng(0)
cur = rng.integers(0, 255, (606, 1077, 3), dtype=np.uint8)
prev = rng.integers(0, 255, (606, 1077, 3), dtype=np.uint8)

results = {}

# --- 双图 ---
print("=== graph=True ===", flush=True)
rt_g = PolicyRuntime(CK, RUN, compile=True, merge=True, graph=True)
print("[t] graph_ready=%s" % rt_g.graph_ready, flush=True)
if rt_g.dual is not None:
    print("[t] dual stats: %s" % json.dumps(rt_g.dual.stats(), ensure_ascii=False), flush=True)

import torch
def bench(rt, n=20):
    ts = []
    with torch.no_grad():
        rt.act(cur, prev, (0.0, 0.8), (0.0, 0.0))
        for _ in range(n):
            torch.xpu.synchronize(); t = time.perf_counter()
            rt.act(cur, prev, (0.0, 0.8), (0.0, 0.0))
            torch.xpu.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return dict(p50=round(ts[len(ts)//2], 1), p95=round(ts[int(len(ts)*0.95)], 1), min=round(ts[0], 1))

results["graph"] = bench(rt_g)
out_g = rt_g.act(cur, prev, (0.0, 0.8), (0.0, 0.0))
print("[t] graph p50=%.1f" % results["graph"]["p50"], flush=True)

del rt_g
import gc; gc.collect(); torch.xpu.empty_cache()

# --- eager compile ---
print("=== graph=False ===", flush=True)
rt_e = PolicyRuntime(CK, RUN, compile=True, merge=True, graph=False)
results["eager"] = bench(rt_e)
out_e = rt_e.act(cur, prev, (0.0, 0.8), (0.0, 0.0))
print("[t] eager p50=%.1f" % results["eager"]["p50"], flush=True)

# --- 正确性 ---
a = np.array(out_g["block"], np.float32)
b = np.array(out_e["block"], np.float32)
results["cos"] = round(float(np.dot(a.ravel(), b.ravel()) /
                             (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)), 8)
results["max_abs"] = round(float(np.abs(a - b).max()), 8)
results["speedup"] = round(results["eager"]["p50"] / results["graph"]["p50"], 2)
print("RESULT " + json.dumps(results, ensure_ascii=False), flush=True)
(ROOT / ".tmp" / "dryrun_20261101" / "e2e_graph.json").write_text(
    json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")