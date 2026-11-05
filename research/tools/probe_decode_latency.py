# -*- coding: utf-8 -*-
"""测单步 decode 延迟 —— 决定「回到自回归行动生成」到底可不可行。

为什么必须先测这个（2026-11-02）：GameScaling 官方实时是 0.13-0.69 s / 步（一步=一个动作）。
我们的控制环是 10 Hz（0.1 s/步），而现在是「一次前向出 12 步」。若要回到原生自回归，
一个 12 步块就需要 12 个 decode 步 —— 必须先知道一步多少钱。

三个口径一起测，避免拿错数：
  1. **prefill**：图像 + 记忆卡 一次前向（我们现在的路径，作对照）；
  2. **decode（带 LM head）**：真实自回归必付的代价，含 248k 词表的投影；
  3. **decode（不投影）**：只跑主干、跳过 LM head —— 用来分离「主干 vs 词表」谁贵。

后者很关键：我们现在**故意绕过 LM head**（train_policy_lora.forward_batch 的注释写了
这是为了省 248k 词表的 logits）。若词表投影占 decode 的大头，那自回归的代价主要不在模型。

用法::

    .venv_xpu/Scripts/python.exe research/tools/probe_decode_latency.py \
        --run .venv_xpu/runs/m0_s3_448/best --tokens 32 --out .venv_xpu/runs/decode_lat
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pct(x, p):
    return float(np.percentile(np.asarray(x, float), p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--run", default=".venv_xpu/runs/m0_s3_448/best")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoProcessor
    from train_policy_lora import PolicyDataset, build_index, load_model

    device = "xpu" if torch.xpu.is_available() else "cpu"
    sync = (lambda: torch.xpu.synchronize()) if device == "xpu" else (lambda: None)

    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = 448 * 256
    samples = [s for s in build_index(Path(a.dataset), None) if s["split"] == "val"][:1]
    ds = PolicyDataset(samples, processor, min_px, max_px, prompt_format="legacy")
    item = ds[0]
    inputs = processor(text=[item["text"]], videos=[item["video"]], return_tensors="pt",
                       padding=True, min_pixels=min_px, max_pixels=max_px,
                       cap_pixels_per_frame=False, do_sample_frames=False)
    kw = {k: v.to(device) for k, v in inputs.items()}

    model, head, hidden = load_model(a.model, 128, device)
    try:
        model.load_adapter(str(Path(a.run) / "adapter"), adapter_name="trained",
                           is_trainable=False)
        model.set_adapter("trained")
    except Exception as e:  # noqa: BLE001
        print("[warn] adapter 没载上：%r" % (e,), flush=True)
    model.eval()
    lm_head = model.get_output_embeddings()
    vocab = int(lm_head.weight.shape[0])
    print("[model] hidden=%d vocab=%d device=%s" % (hidden, vocab, device), flush=True)

    def prefill():
        sync(); t = time.perf_counter()
        with torch.no_grad():
            out = model(**kw, use_cache=True)
        sync(); return out, time.perf_counter() - t

    # 预热
    for _ in range(3):
        prefill()
    pre_ms = []
    for _ in range(5):
        _, dt = prefill()
        pre_ms.append(dt * 1000)

    # ---- decode ----
    out, _ = prefill()
    past = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1:, :], dim=-1)
    dec_full, dec_trunk = [], []
    with torch.no_grad():
        for i in range(a.tokens):
            sync(); t = time.perf_counter()
            o = model(input_ids=nxt, past_key_values=past, use_cache=True)
            logits = o.logits[:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            past = o.past_key_values
            sync(); dec_full.append((time.perf_counter() - t) * 1000)
        # 只跑主干一遍（不投影词表）：与上面同一次输入形状
        for i in range(min(8, a.tokens)):
            sync(); t = time.perf_counter()
            o = model.base_model.model.model(input_ids=nxt, past_key_values=past,
                                             use_cache=True)
            past = o.past_key_values
            sync(); dec_trunk.append((time.perf_counter() - t) * 1000)

    res = dict(
        device=device, vocab=vocab, hidden=hidden, tokens=a.tokens,
        prefill_ms=dict(p50=pct(pre_ms, 50), p95=pct(pre_ms, 95)),
        decode_full_ms=dict(p50=pct(dec_full, 50), p95=pct(dec_full, 95)),
        decode_trunk_ms=dict(p50=pct(dec_trunk, 50), p95=pct(dec_trunk, 95)),
    )
    blk = 24   # 12 步 x 2 值 = 一个动作块的 token 数（粗估）
    res["block_24tok_full_ms"] = res["decode_full_ms"]["p50"] * blk
    res["block_24tok_trunk_ms"] = res["decode_trunk_ms"]["p50"] * blk
    res["fits_2hz"] = bool(res["block_24tok_full_ms"] <= 500)
    res["fits_1hz"] = bool(res["block_24tok_full_ms"] <= 1000)

    print()
    print("  prefill(图像+卡)     p50=%.1f ms  p95=%.1f ms"
          % (res["prefill_ms"]["p50"], res["prefill_ms"]["p95"]))
    print("  decode 带 LM head    p50=%.1f ms  p95=%.1f ms"
          % (res["decode_full_ms"]["p50"], res["decode_full_ms"]["p95"]))
    print("  decode 跳过词表投影  p50=%.1f ms  p95=%.1f ms"
          % (res["decode_trunk_ms"]["p50"], res["decode_trunk_ms"]["p95"]))
    print("  ⇒ 24 token 一块：带词表 %.0f ms / 跳词表 %.0f ms"
          % (res["block_24tok_full_ms"], res["block_24tok_trunk_ms"]))
    print("  ⇒ 2 Hz(500ms) %s   1 Hz(1000ms) %s"
          % ("可行" if res["fits_2hz"] else "不可行",
             "可行" if res["fits_1hz"] else "不可行"))
    print("  对照：现在的「一次前向出 12 步」= prefill %.1f ms" % res["prefill_ms"]["p50"])

    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "decode_latency.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (o / "decode_latency.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
