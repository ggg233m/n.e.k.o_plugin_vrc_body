# -*- coding: utf-8 -*-
"""静态形状（dynamic=False）能不能编过 decode —— 检验「fp64 标量来自 dynamic SymFloat」假设。

2026-11-03 实测链：
  decode eager 6258 个算子 / 338 ms；想用 Inductor 融合，挂在
  ZE_RESULT_ERROR_MODULE_BUILD_FAILURE（IGC 建 module 失败）。
  开 SYCL_UR_USE_LEVEL_ZERO_V2=1 **无效**（同一 kernel 同一错）。
  翻生成的 Triton 源码发现签名里有 **fp64 标量**（in_ptr2: fp64），
  而设备属性写着 has_fp64: False —— A770(dg2) 没有 FP64。

  假设 H1：dynamic=True 时形状推导出的 SymFloat 成为 kernel 标量参数，
           Python float -> fp64 -> IGC 拒绝。改 dynamic=False 应能编过。
  假设 H2：transformers 的 rope attention_scaling 是 float -> fp64。
           用 --patch-rope 把缩放显式转 fp32。

静态形状下 KV cache 每步增长会触发重编，所以本脚本**只测一个形状**：
  prefill 一次得到 (nxt0, past0)，之后用同一组输入反复调用。
  这足以回答「能不能编出来 / 算子数掉到多少」，不追求语义正确。

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_decode_static_compile.py \
        --out .venv_xpu/runs/decode_static [--patch-rope]
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


def _p50(xs):
    return float(np.percentile(np.asarray(xs, float), 50))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--run", default=".venv_xpu/runs/m0_s3_448/best")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--patch-rope", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    import torch._inductor.config as ic
    ic.max_autotune = False
    ic.max_autotune_gemm = False
    ic.coordinate_descent_tuning = False
    try:
        ic.compile_threads = 1
    except Exception:  # noqa: BLE001
        pass

    dev = "xpu" if torch.xpu.is_available() else "cpu"
    sync = (lambda: torch.xpu.synchronize()) if dev == "xpu" else (lambda: None)

    if a.patch_rope:
        import transformers.models.qwen3_5.modeling_qwen3_5 as M

        def _fwd(self, x, position_ids):
            inv = self.inv_freq.to(device=x.device, dtype=torch.float32)
            freqs = position_ids[..., None].float() * inv
            sc = torch.as_tensor(float(self.attention_scaling),
                                 dtype=torch.float32, device=x.device)
            return (freqs.cos() * sc).to(x.dtype), (freqs.sin() * sc).to(x.dtype)

        M.Qwen3_5TextRotaryEmbedding.forward = _fwd
        print("[rope] attention_scaling 强制 fp32", flush=True)

    from transformers import AutoProcessor
    from train_policy_lora import PolicyDataset, build_index, load_model
    from torch.profiler import ProfilerActivity, profile

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

    with torch.no_grad():
        for _ in range(3):
            o = model(**kw, use_cache=True)
        sync()
        past0 = o.past_key_values
        nxt0 = torch.argmax(o.logits[:, -1:, :], dim=-1)
        del o
    print("[prefill] past len=%s" % (len(past0),), flush=True)

    tgt = model.base_model.model.model if hasattr(model, "base_model") else model
    print("[compile] dynamic=False ...", flush=True)
    t0 = time.perf_counter()
    comp = torch.compile(tgt, dynamic=False, fullgraph=False, mode="default")
    model.base_model.model.model = comp

    res = {"device": dev, "patch_rope": bool(a.patch_rope), "dynamic": False}
    walls = []
    try:
        with torch.no_grad():
            for i in range(3):
                sync()
                tt = time.perf_counter()
                model(input_ids=nxt0, past_key_values=past0, use_cache=True)
                sync()
                print("  warm%d %.0f ms" % (i, (time.perf_counter() - tt) * 1000), flush=True)
            for _ in range(a.reps):
                sync()
                tt = time.perf_counter()
                model(input_ids=nxt0, past_key_values=past0, use_cache=True)
                sync()
                walls.append((time.perf_counter() - tt) * 1000)
        print("[compile] 总耗时 %.0f s" % (time.perf_counter() - t0), flush=True)

        act = [ProfilerActivity.CPU]
        if dev == "xpu":
            act.append(ProfilerActivity.XPU)
        with profile(activities=act) as prof:
            model(input_ids=nxt0, past_key_values=past0, use_cache=True)
            sync()
        rows = []
        gpu_us = 0.0
        for e in prof.key_averages():
            dtt = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
            if dtt <= 0:
                continue
            gpu_us += dtt
            rows.append((int(e.count), str(e.key), dtt / 1000.0))
        rows.sort(reverse=True)
        res["static"] = dict(
            wall_ms=_p50(walls), wall_all=[round(w, 1) for w in walls],
            kernels=sum(r[0] for r in rows), gpu_ms=gpu_us / 1000.0,
            top=[{"n": n, "key": k, "gpu_ms": round(g, 1)} for n, k, g in rows[:12]])
        s = res["static"]
        res["block_24tok_ms"] = s["wall_ms"] * 24
        res["fits_2hz"] = bool(s["wall_ms"] * 24 <= 500)
        print()
        print("[S] wall %.1f ms | kernel %d | GPU %.1f ms"
              % (s["wall_ms"], s["kernels"], s["gpu_ms"]), flush=True)
        for r in s["top"][:10]:
            print("      %-50s n=%-5d %.1f ms" % (r["key"][:50], r["n"], r["gpu_ms"]), flush=True)
        print("  => 24 token 一块 %.0f ms | 2Hz %s"
              % (res["block_24tok_ms"], "可行" if res["fits_2hz"] else "不可行"), flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        res["static_error"] = repr(e)
        print("[S] 失败：%r" % (e,), flush=True)

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    tag = "rope" if a.patch_rope else "plain"
    (o / ("decode_static_%s.json" % tag)).write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
