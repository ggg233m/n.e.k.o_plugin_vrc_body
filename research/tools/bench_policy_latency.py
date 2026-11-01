# -*- coding: utf-8 -*-
"""实测 policy 推理延迟（G1 用）：加载 base + LoRA adapter + head，真实 448×256 样本。

输出：prefill p50/p95/p99（eager，bf16，batch=1，use_cache=False）、
图像预处理（2 张 JPEG 解码 + chat template + processor）耗时、峰值显存、token 数。

用法::

    G:\\vrc_recording\\vrc_model\\venv_xpu\\Scripts\\python.exe research/tools/bench_policy_latency.py \\
        --run h:\\...\\.venv_xpu\\runs\\m0_s3_448\\best \\
        --out G:\\vrc_recording\\vrc_model\\runs\\latency_448.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_policy_lora import (  # noqa: E402
    PolicyDataset, build_index, forward_batch, load_model, make_collate,
)


def pct(xs, p):
    return float(np.percentile(np.array(xs), p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3")
    ap.add_argument("--run", default=r"h:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\.venv_xpu\runs\m0_s3_448\best")
    ap.add_argument("--dataset", default=r"G:\vrc_recording\vrc_model\dataset\m0_policy_448")
    ap.add_argument("--size", nargs=2, type=int, default=[448, 256])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--compile", action="store_true", help="torch.compile 内层 Qwen3_5Model")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = int(a.size[0]) * int(a.size[1])

    samples = build_index(Path(a.dataset), None)
    val = [s for s in samples if s["split"] == "val"]
    ds = PolicyDataset(val[:1], processor, min_px, max_px)
    collate = make_collate(processor, min_px, max_px)
    sample = ds[0]
    v = sample["video"]
    if v.shape[1] != int(a.size[1]) or v.shape[2] != int(a.size[0]):
        # 视频分支不吃 min/max_pixels，必须先把帧缩到目标尺寸，否则测的还是原图
        sample["video"] = np.stack([
            np.asarray(Image.fromarray(f).resize((int(a.size[0]), int(a.size[1])), Image.BICUBIC))
            for f in v])
    batch = collate([sample])

    # 预处理耗时：JPEG 解码 + chat template + processor
    def prep():
        b = collate([ds[0]])
        return b
    for _ in range(3):
        prep()
    t_pre = []
    for _ in range(20):
        t = time.perf_counter(); prep(); t_pre.append(time.perf_counter() - t)

    model, head, hidden = load_model(a.model, 128, device)
    model.load_adapter(str(Path(a.run) / "adapter"), adapter_name="trained", is_trainable=False)
    model.set_adapter("trained")
    head.load_state_dict(torch.load(Path(a.run) / "head.pt", map_location=device))
    model.eval(); head.eval()
    if a.compile:
        print("[compile] torch.compile inner Qwen3_5Model …", flush=True)
        model.base_model.model.model = torch.compile(model.base_model.model.model)

    inputs = batch["inputs"]
    if device == "xpu":
        torch.xpu.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(a.warmup):
            forward_batch(model, head, inputs, device)
        if device == "xpu":
            torch.xpu.synchronize()
        times = []
        for _ in range(a.iters):
            if device == "xpu":
                torch.xpu.synchronize()
            t = time.perf_counter()
            forward_batch(model, head, inputs, device)
            if device == "xpu":
                torch.xpu.synchronize()
            times.append(time.perf_counter() - t)

    peak = (torch.xpu.max_memory_allocated() / 2 ** 30) if device == "xpu" else None
    res = dict(
        device=device, iters=a.iters, size=list(a.size), compiled=bool(a.compile),
        input_ids=list(inputs["input_ids"].shape),
        grid=inputs["video_grid_thw"].tolist() if "video_grid_thw" in inputs else None,
        prefill_ms=dict(p50=pct(times, 50) * 1000, p95=pct(times, 95) * 1000,
                        p99=pct(times, 99) * 1000, max=max(times) * 1000),
        preprocess_ms=dict(p50=pct(t_pre, 50) * 1000, p95=pct(t_pre, 95) * 1000),
        peak_gib=round(peak, 2) if peak else None,
        budget_note="2 Hz 要求 p95 <= 200 ms；1 Hz 要求 <= 1000 ms",
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
