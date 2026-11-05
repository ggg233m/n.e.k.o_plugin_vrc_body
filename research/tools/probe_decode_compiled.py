# -*- coding: utf-8 -*-
"""decode 在 torch.compile 下能不能把 6265 个算子融掉 —— 判决「自回归可行/不可行」。

背景（2026-11-02 实测）：
  decode 一个 token = wall 387 ms，其中 **6265 个 kernel**，GPU 忙碌只 88.5 ms。
  单次算子固定开销 47.8 us，6265 x 47.8 us = 300 ms —— 时间几乎全花在这。
  而 0.896B BF16 权重 1.79 GB / 446 GB/s = 4.0 ms 才是理论下限。

  6265 个算子的来源不是硬件，是 transformers 的三行警告：
    causal_conv1d / chunk_gated_delta_rule / fused_recurrent_gated_delta_rule 都没装，
    18 层 GatedDeltaNet 退化成参考 PyTorch 实现，每层炸成几百个微算子。

所以问题变成：**Inductor 能不能把参考实现融掉**。本脚本直接数 kernel 个数。

三个口径：
  A. eager（基线，应复现 6265 / 387 ms）
  B. compile(dynamic=True)  —— KV cache 长度每步在变，必须开 dynamic 否则每步重编
  C. compile(mode='reduce-overhead') —— 若能上图，进一步省掉提交往返

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_decode_compiled.py \
        --out .venv_xpu/runs/decode_compiled --mode default
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

# ⚠️ Inductor 编 Triton kernel 时 Triton 的 intel driver 会惰性调 C 编译器；
# 不把 oneAPI 塞进 PATH 就直接 "Failed to find C compiler"（2026-11-02 踩过两次）。
_ONEAPI_BIN = r"D:\Program Files (x86)\Intel\oneAPI\2026.0\bin"
if os.path.isdir(_ONEAPI_BIN):
    os.environ["PATH"] = _ONEAPI_BIN + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("ONEAPI_ROOT", r"D:\Program Files (x86)\Intel\oneAPI")
    os.environ.setdefault("CC", "icpx")
    os.environ.setdefault("CXX", "icpx")
else:
    print("[warn] oneAPI bin 不存在，compile 大概率失败", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _p50(xs):
    return float(np.percentile(np.asarray(xs, float), 50))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--run", default=".venv_xpu/runs/m0_s3_448/best")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=6, help="编译后连跑几个 token 看稳定性")
    ap.add_argument("--mode", default="default", choices=["default", "reduce-overhead", "max-autotune-no-cudagraphs"])
    ap.add_argument("--skip-eager", action="store_true")
    ap.add_argument("--specialize-float", action="store_true",
                    help="dynamo 把 Python float 折成常量，避免 fp64 kernel 标量参数")
    ap.add_argument("--patch-rope", action="store_true",
                    help="把 rope 的 attention_scaling 强制成 fp32 张量，去掉图里的 fp64 标量")
    ap.add_argument("--eager-ref", default=".venv_xpu/runs/launch_ovh/launch_overhead.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch

    dev = "xpu" if torch.xpu.is_available() else "cpu"
    sync = (lambda: torch.xpu.synchronize()) if dev == "xpu" else (lambda: None)

    if a.patch_rope:
        # 只把 attention_scaling 这个 Python float 换成 fp32 张量，逻辑一行不动。
        # 原实现：attention_factor = 1.0（float64）；cos = freqs.cos() * self.attention_scaling
        # Inductor 把它当 kernel 标量参数 -> 'fp64' -> IGC「Double type is not supported」。
        # 注意 Qwen3.5 是 mrope（position_ids expand 到 (3,bs,pos)），不能整体替换 forward。
        import transformers.models.qwen3_5.modeling_qwen3_5 as _M

        _orig_fwd = _M.Qwen3_5TextRotaryEmbedding.forward

        def _rope_fwd(self, x, position_ids):
            sc = getattr(self, "attention_scaling", None)
            swapped = isinstance(sc, float)
            if swapped:
                self.attention_scaling = torch.as_tensor(
                    sc, dtype=torch.float32, device=x.device)
            try:
                return _orig_fwd(self, x, position_ids)
            finally:
                if swapped:
                    self.attention_scaling = sc

        _M.Qwen3_5TextRotaryEmbedding.forward = _rope_fwd
        print("[rope] attention_scaling 换成 fp32 张量（去 fp64 标量）", flush=True)

    if a.specialize_float:
        torch._dynamo.config.specialize_float = True
        print("[dynamo] specialize_float=True（Python float 折成常量，去 fp64 标量）",
              flush=True)

    import torch._inductor.config as ic
    ic.max_autotune = False
    ic.max_autotune_gemm = False
    ic.coordinate_descent_tuning = False
    # Windows 上 compile_threads > 1 会抛 "pass_fds not supported"，必须钉死 1
    try:
        ic.compile_threads = 1
    except Exception:  # noqa: BLE001
        pass

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
    except Exception as e:  # noqa: BLE001
        print("[warn] adapter: %r" % (e,), flush=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print("[model] %.3f B params | device=%s | torch=%s"
          % (n_params / 1e9, dev, torch.__version__), flush=True)

    with torch.no_grad():
        for _ in range(3):
            o = model(**kw, use_cache=True)
        sync()
        past0 = o.past_key_values
        nxt0 = torch.argmax(o.logits[:, -1:, :], dim=-1)
        del o

    def decode_measure(tag, reps=None):
        reps = reps or a.reps
        walls, past, nxt = [], past0, nxt0
        with torch.no_grad():
            for i in range(2):     # warm（compile 在这里触发）
                t0 = time.perf_counter()
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                nxt = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
                past = o.past_key_values
                sync()
                print("  [%s] warm%d %.0f ms" % (tag, i, (time.perf_counter() - t0) * 1000),
                      flush=True)
            for i in range(reps):
                sync()
                t0 = time.perf_counter()
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                nxt = torch.argmax(o.logits[:, -1, :], dim=-1, keepdim=True)
                past = o.past_key_values
                sync()
                walls.append((time.perf_counter() - t0) * 1000)

            act = [ProfilerActivity.CPU]
            if dev == "xpu":
                act.append(ProfilerActivity.XPU)
            with profile(activities=act) as prof:
                o = model(input_ids=nxt, past_key_values=past, use_cache=True)
                sync()
            ka = prof.key_averages()
            rows = []
            gpu_us = 0.0
            for e in ka:
                dt = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
                if dt <= 0:
                    continue
                gpu_us += dt
                rows.append((int(e.count), str(e.key), dt / 1000.0))
            rows.sort(reverse=True)
            n_kern = sum(r[0] for r in rows)
        out = dict(wall_ms=_p50(walls), wall_all=[round(w, 1) for w in walls],
                   kernels=n_kern, gpu_ms=gpu_us / 1000.0,
                   gpu_frac=gpu_us / 1000.0 / max(_p50(walls), 1e-9),
                   top=[{"n": n, "key": k, "gpu_ms": round(g, 1)} for n, k, g in rows[:14]])
        return out

    res = dict(model=a.model, params=int(n_params), mode=a.mode, device=dev)
    print()
    print("=" * 70, flush=True)
    if not a.skip_eager:
        print("[A] eager 基线", flush=True)
        t0 = time.perf_counter()
        res["eager"] = decode_measure("eager")
        e = res["eager"]
        print("[A] wall %.1f ms | kernel %d | GPU %.1f ms (%.0f%%)  [%.0fs]"
              % (e["wall_ms"], e["kernels"], e["gpu_ms"], 100 * e["gpu_frac"],
                 time.perf_counter() - t0), flush=True)
        for r in e["top"][:8]:
            print("      %-52s n=%-5d %.1f ms" % (r["key"][:52], r["n"], r["gpu_ms"]), flush=True)

    # ---------------- B/C. compile ----------------
    print()
    print("[B] torch.compile(mode=%s, dynamic=True) —— 首次会编译，耐心等" % a.mode, flush=True)
    t0 = time.perf_counter()
    tgt = model.base_model.model.model if hasattr(model, "base_model") else model
    try:
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.accumulated_cache_size_limit = 256
    except Exception:  # noqa: BLE001
        pass
    compiled = torch.compile(tgt, dynamic=True, fullgraph=False, mode=a.mode)
    model.base_model.model.model = compiled
    try:
        res["compiled"] = decode_measure("compile")
        c = res["compiled"]
        print("[B] wall %.1f ms | kernel %d | GPU %.1f ms (%.0f%%)  [总 %.0fs]"
              % (c["wall_ms"], c["kernels"], c["gpu_ms"], 100 * c["gpu_frac"],
                 time.perf_counter() - t0), flush=True)
        for r in c["top"][:10]:
            print("      %-52s n=%-5d %.1f ms" % (r["key"][:52], r["n"], r["gpu_ms"]), flush=True)
        if "eager" in res:
            e = res["eager"]
            print("  => kernel %d -> %d (%.2fx) | wall %.1f -> %.1f ms (%.1fx)"
                  % (e["kernels"], c["kernels"], e["kernels"] / max(c["kernels"], 1),
                     e["wall_ms"], c["wall_ms"], e["wall_ms"] / max(c["wall_ms"], 1e-9)), flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        res["compiled_error"] = repr(e)

    # ---------------- 判据 ----------------
    best = res.get("compiled") or res.get("eager") or {}
    if best:
        per = best.get("wall_ms", 1e9)
        res["block_24tok_ms"] = per * 24
        res["fits_2hz"] = bool(per * 24 <= 500)
        res["fits_1hz"] = bool(per * 24 <= 1000)
        print()
        print("  单 token %.1f ms -> 24 token 一块 %.0f ms | 2Hz %s | 1Hz %s"
              % (per, per * 24,
                 "可行" if res["fits_2hz"] else "不可行",
                 "可行" if res["fits_1hz"] else "不可行"), flush=True)

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / ("decode_compiled_%s.json" % a.mode.replace("-", "_"))).write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
