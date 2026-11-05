# -*- coding: utf-8 -*-
"""融合后的 policy 前向（prefill）延迟 —— 和 G1 的 123.4 ms 直接对比。

背景：2026-11-03 发现 decode 编不过的真因是 fp64（Python float 被当运行期标量参数），
两处修补后 decode 图能编译且算子数 6258 -> 1824。
而 G1 的 policy 前向目前是「手工 XPUGraph，未融合」= p95 123.4 ms / p50 ~88 ms。
本脚本量同一口径下「融合但不上图」是多少，用来判断融合能不能和上图叠加。

口径：448x256 图像 + 记忆卡，单次前向，无 KV cache。

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_policy_fused.py --out .venv_xpu/runs/policy_fused
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_REPO / ".tc_cache" / "inductor"))
os.environ.setdefault("PYTHONUTF8", "1")

_ONEAPI_BIN = r"D:\Program Files (x86)\Intel\oneAPI\2026.0\bin"
if os.path.isdir(_ONEAPI_BIN):
    os.environ["PATH"] = _ONEAPI_BIN + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("ONEAPI_ROOT", r"D:\Program Files (x86)\Intel\oneAPI")
    os.environ.setdefault("CC", "icpx")
    os.environ.setdefault("CXX", "icpx")

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _p(xs, q):
    return float(np.percentile(np.asarray(xs, float), q))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--run", default=".venv_xpu/runs/m0_s3_448/best")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--dynamic", action="store_true", help="compile 用 dynamic=True")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    torch._dynamo.config.specialize_float = True

    import transformers.models.qwen3_5.modeling_qwen3_5 as _M
    _orig_fwd = _M.Qwen3_5TextRotaryEmbedding.forward

    def _rope_fwd(self, x, position_ids):
        sc = getattr(self, "attention_scaling", None)
        swapped = isinstance(sc, float)
        if swapped:
            self.attention_scaling = torch.as_tensor(sc, dtype=torch.float32, device=x.device)
        try:
            return _orig_fwd(self, x, position_ids)
        finally:
            if swapped:
                self.attention_scaling = sc

    _M.Qwen3_5TextRotaryEmbedding.forward = _rope_fwd
    print("[patch] specialize_float=True + rope attention_scaling fp32", flush=True)

    import torch._inductor.config as ic
    ic.max_autotune = False
    ic.max_autotune_gemm = False
    ic.coordinate_descent_tuning = False
    try:
        ic.compile_threads = 1
    except Exception:  # noqa: BLE001
        pass

    from transformers import AutoProcessor
    from train_policy_lora import PolicyDataset, build_index, load_model
    from torch.profiler import ProfilerActivity, profile

    dev = "xpu" if torch.xpu.is_available() else "cpu"
    sync = (lambda: torch.xpu.synchronize()) if dev == "xpu" else (lambda: None)

    processor = AutoProcessor.from_pretrained(a.model)
    mn = mx = 448 * 256
    samples = [s for s in build_index(Path(a.dataset), None) if s["split"] == "val"][:1]
    ds = PolicyDataset(samples, processor, mn, mx, prompt_format="legacy")
    it = ds[0]
    inputs = processor(text=[it["text"]], videos=[it["video"]], return_tensors="pt",
                       padding=True, min_pixels=mn, max_pixels=mx,
                       cap_pixels_per_frame=False, do_sample_frames=False)
    kw = {k: v.to(dev) for k, v in inputs.items()}

    model, head, hidden = load_model(a.model, 128, dev)
    try:
        model.load_adapter(str(Path(a.run) / "adapter"), adapter_name="trained",
                           is_trainable=False)
        model.set_adapter("trained")
    except Exception:  # noqa: BLE001
        pass
    model.eval()
    print("[model] %.3f B params | device=%s" % (
        sum(p.numel() for p in model.parameters()) / 1e9, dev), flush=True)

    def measure(tag, reps):
        walls = []
        with torch.no_grad():
            for _ in range(2):
                model(**kw)
            sync()
            for _ in range(reps):
                sync()
                t0 = time.perf_counter()
                model(**kw)
                sync()
                walls.append((time.perf_counter() - t0) * 1000)
            act = [ProfilerActivity.CPU]
            if dev == "xpu":
                act.append(ProfilerActivity.XPU)
            with profile(activities=act) as prof:
                model(**kw)
                sync()
        rows, gpu_us = [], 0.0
        for e in prof.key_averages():
            dt = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
            if dt <= 0:
                continue
            gpu_us += dt
            rows.append((int(e.count), str(e.key), dt / 1000.0))
        rows.sort(reverse=True)
        r = dict(tag=tag, n=len(walls),
                 p50=_p(walls, 50), p95=_p(walls, 95), mn=float(min(walls)),
                 kernels=sum(x[0] for x in rows), gpu_ms=gpu_us / 1000.0,
                 top=[{"n": x[0], "key": x[1][:60], "gpu_ms": round(x[2], 1)} for x in rows[:8]])
        print("  [%s] p50=%.1f ms p95=%.1f ms min=%.1f | kernel %d | GPU %.1f ms"
              % (tag, r["p50"], r["p95"], r["mn"], r["kernels"], r["gpu_ms"]), flush=True)
        return r

    res = {"device": dev, "dynamic": bool(a.dynamic)}
    print("[A] eager 基线", flush=True)
    res["eager"] = measure("eager", a.iters)

    print("[B] compile（融合）...", flush=True)
    t0 = time.perf_counter()
    model = torch.compile(model, dynamic=a.dynamic, fullgraph=False, mode="default")
    try:
        res["compiled"] = measure("compiled", a.iters)
        print("  [compile] 总耗时 %.0f s" % (time.perf_counter() - t0), flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        res["compiled_error"] = repr(e)

    e, c = res.get("eager"), res.get("compiled")
    if e and c:
        print("  => 算子 %d -> %d (%.2fx) | p50 %.1f -> %.1f ms (%.2fx) | p95 %.1f -> %.1f"
              % (e["kernels"], c["kernels"], e["kernels"] / max(c["kernels"], 1),
                 e["p50"], c["p50"], e["p50"] / max(c["p50"], 1e-9), e["p95"], c["p95"]), flush=True)
        print("  => 对照 G1 现状（XPUGraph 未融合）：p95 123.4 ms / 并发下 175.8 ms", flush=True)

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "policy_fused.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print("[done] %s" % o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
