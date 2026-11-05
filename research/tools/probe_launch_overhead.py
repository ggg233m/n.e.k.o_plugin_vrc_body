# -*- coding: utf-8 -*-
"""定位「每次算子调用的固定开销」——为什么 llama.cpp/ComfyUI 跑大模型不慢，我们 decode 慢一百倍。

墙钟 = 固定开销 a + 计算 b·工作量。
  a 与 kernel 大小基本无关（Python -> dispatcher -> SYCL -> Level Zero -> GPU）
  b·工作量 才随规模涨
  => 同样的 a，工作量越大 a 占比越小。这是唯一的关键。

四段测量，从便宜到贵：
  1. 极小算子 enqueue vs total —— 提交是异步排队还是阻塞往返；
  2. **同样 kernel 个数、128 倍工作量** —— 若两者耗时接近，就证明 a 主导；
  3. matmul 规模扫描拟合 t = a + b·S²；再量一次实测显存带宽，算 decode 的理论下限；
  4. **profile 一个 decode step**：kernel 个数 / GPU 忙碌时间 / 墙钟，直接看 GPU 空闲占比；
  5. llama.cpp / ollama 是否存在，能真机对比最好。

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_launch_overhead.py --out .venv_xpu/runs/launch_ovh
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _fit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    A = np.stack([np.ones_like(x), x], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else 0.0
    return float(coef[0]), float(coef[1]), r2


def _loop(fn, n, warm, sync):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    enq = (time.perf_counter() - t0) / n
    sync()
    tot = (time.perf_counter() - t0) / n
    return enq, tot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--run", default=".venv_xpu/runs/m0_s3_448/best")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--micro-only", action="store_true",
                    help="只跑微基准，不加载模型（用来对比环境变量）")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch

    dev = "xpu" if torch.xpu.is_available() else "cpu"
    sync = (lambda: torch.xpu.synchronize()) if dev == "xpu" else (lambda: None)
    res = {"device": dev}
    print("[dev] %s  torch=%s" % (dev, torch.__version__), flush=True)

    # ---------- 1. 极小算子：enqueue vs total ----------
    x1 = torch.ones(1, device=dev)
    enq, tot = _loop(lambda: x1.add_(1.0), 500, 20, sync)
    res["tiny_enqueue_us"] = enq * 1e6
    res["tiny_total_us"] = tot * 1e6
    res["tiny_async"] = bool(enq < tot * 0.5)
    print("[1] 单元素 add_ : enqueue %.1f us  总计 %.1f us  -> %s"
          % (enq * 1e6, tot * 1e6,
             "异步排队" if enq < tot * 0.5 else "同步阻塞往返"), flush=True)

    # ---------- 2. 同样一次调用，128 倍工作量 ----------
    S = 1024
    W = torch.randn(S, S, device=dev, dtype=torch.bfloat16)
    h1 = torch.randn(S, 1, device=dev, dtype=torch.bfloat16)
    hN = torch.randn(S, 128, device=dev, dtype=torch.bfloat16)
    _, t1 = _loop(lambda: W @ h1, 60, 10, sync)
    _, tN = _loop(lambda: W @ hN, 60, 10, sync)
    res["gemv_S1024_bs1_us"] = t1 * 1e6
    res["gemv_S1024_bs128_us"] = tN * 1e6
    res["work_ratio"] = 128.0
    res["time_ratio"] = (tN / t1) if t1 > 0 else None
    print("[2] W(1024x1024) @ h : bs=1 %.0f us   bs=128 %.0f us"
          % (t1 * 1e6, tN * 1e6), flush=True)
    print("    -> 工作量 x%.0f，耗时只 x%.1f  => 固定开销占 bs=1 的 %.0f%%"
          % (128.0, (tN / t1) if t1 > 0 else 0,
             100.0 * (1.0 - (tN - t1) / max(t1, 1e-9)) if t1 > 0 else 0), flush=True)

    # ---------- 3. 规模扫描 + 实测带宽 ----------
    sizes = [128, 256, 512, 1024, 2048, 4096]
    ts, ws = [], []
    for sz in sizes:
        W2 = torch.randn(sz, sz, device=dev, dtype=torch.bfloat16)
        h2 = torch.randn(sz, 1, device=dev, dtype=torch.bfloat16)
        n = 40 if sz <= 2048 else 15
        _, tt = _loop(lambda: W2 @ h2, n, 5, sync)
        ts.append(tt * 1e6)
        ws.append(sz * sz * 2.0 / n if False else float(sz * sz))
        print("    S=%-5d t=%9.1f us" % (sz, tt * 1e6), flush=True)
        del W2, h2
    fa, fb, r2 = _fit(ws, ts)
    res["fit_a_us"] = fa
    res["fit_b_us_per_word"] = fb
    res["fit_r2"] = r2
    print("[3] 拟合 t = a + b*S^2 :  a=%.1f us   b=%.3e us  r2=%.4f" % (fa, fb, r2), flush=True)

    big = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=dev)   # 256 MB
    dst = torch.empty_like(big)
    _, tcp = _loop(lambda: dst.copy_(big), 20, 5, sync)
    bw = 2.0 * big.numel() * 4.0 / tcp / 1e9
    res["measured_bw_gbs"] = bw
    print("    实测 memcpy 带宽 %.0f GB/s（256 MB 对拷）" % bw, flush=True)
    del big, dst

    # ---------- 4. CPU vs XPU 同一算子 ----------
    try:
        Sc = 1024
        Wc_x = torch.randn(Sc, Sc, dtype=torch.bfloat16)
        hc = torch.randn(Sc, 1, dtype=torch.bfloat16)
        _, t_cpu = _loop(lambda: Wc_x @ hc, 200, 10, lambda: None)
        res["cpu_gemv_us"] = t_cpu * 1e6
        print("[4] 同一 GEMV：CPU %.1f us  vs %s %.1f us"
              % (t_cpu * 1e6, dev, t1 * 1e6), flush=True)
    except Exception as e:  # noqa: BLE001
        print("[4] CPU 对照失败：%r" % (e,), flush=True)

    if a.micro_only:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "launch_overhead.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print("[done-micro] %s" % (o / "launch_overhead.json"))
        return 0

    # ---------- 5. profile 一个 decode step ----------
    try:
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
        n_params = sum(p.numel() for p in model.parameters())
        res["params"] = int(n_params)
        print("[5] 模型参数 %.3f B，BF16 权重 %.2f GB"
              % (n_params / 1e9, n_params * 2 / 1e9), flush=True)

        with torch.no_grad():
            for _ in range(3):
                o = model(**kw, use_cache=True)
            sync()
            past = o.past_key_values
            nxt = torch.argmax(o.logits[:, -1:, :], dim=-1)
            sync()

            # 干净墙钟
            walls = []
            for _ in range(5):
                sync()
                t0 = time.perf_counter()
                oo = model(input_ids=nxt, past_key_values=past, use_cache=True)
                nxt2 = torch.argmax(oo.logits[:, -1, :], dim=-1, keepdim=True)
                sync()
                walls.append((time.perf_counter() - t0) * 1000)
                past = oo.past_key_values
                nxt = nxt2
            wall_p50 = float(np.percentile(walls, 50))
            res["decode_wall_ms"] = wall_p50

            # profile 一次
            act = [ProfilerActivity.CPU]
            if dev == "xpu":
                act.append(ProfilerActivity.XPU)
            with profile(activities=act) as prof:
                oo = model(input_ids=nxt, past_key_values=past, use_cache=True)
                _ = torch.argmax(oo.logits[:, -1, :], dim=-1, keepdim=True)
                sync()
            ka = prof.key_averages()
            n_kern = sum(int(e.count) for e in ka if getattr(e, "self_device_time_total", 0) > 0)
            dev_us = sum(float(getattr(e, "self_device_time_total", 0.0)) for e in ka)
            cpu_us = sum(float(getattr(e, "self_cpu_time_total", 0.0)) for e in ka)
            res["decode_kernels"] = n_kern
            res["decode_gpu_busy_ms"] = dev_us / 1000.0
            res["decode_cpu_time_ms"] = cpu_us / 1000.0
            res["gpu_busy_frac"] = dev_us / 1000.0 / max(wall_p50, 1e-9)
            print("    decode 一次：wall %.1f ms | kernel 个数 %d | GPU 忙碌 %.1f ms (%.1f%%) "
                  "| CPU 自耗时 %.1f ms"
                  % (wall_p50, n_kern, dev_us / 1000.0,
                     100.0 * dev_us / 1000.0 / max(wall_p50, 1e-9), cpu_us / 1000.0), flush=True)

            wbytes = n_params * 2.0
            res["decode_floor_ms"] = wbytes / (bw * 1e9) * 1000.0
            res["decode_vs_floor"] = wall_p50 / max(wall_p50 * 0 + res["decode_floor_ms"], 1e-9)
            print("    带宽下限：读一遍权重 %.2f GB / %.0f GB/s = %.2f ms  => 实测是下限的 %.0f 倍"
                  % (wbytes / 1e9, bw, res["decode_floor_ms"], res["decode_vs_floor"]), flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        res["profile_error"] = repr(e)

    # ---------- 6. llama.cpp / ollama ----------
    found = {}
    for name in ("llama-cli", "llama-server", "main", "ollama", "lms", "llama-bench"):
        p = shutil.which(name)
        if p:
            found[name] = p
    res["llama_tools"] = found
    print("[6] 本机 llama.cpp/ollama 类工具：%s"
          % (", ".join(found) if found else "未找到"), flush=True)

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "launch_overhead.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (o / "launch_overhead.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
