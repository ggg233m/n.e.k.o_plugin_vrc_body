# -*- coding: utf-8 -*-
"""G2 一致性：torch.compile 与 eager 在同一批真实样本上的 hidden / 动作输出对比。

关：pooled hidden 余弦（方案 §6 门 cos >= 0.999）、动作预测 cos / max|Δ| / MSE。
用法（必须带 oneAPI 环境，否则 compile 退回 eager）::

    set PATH += D:\\Program Files (x86)\\Intel\\oneAPI\\2026.0\\bin
    set ONEAPI_ROOT=...; set CC=icpx; set CXX=icpx
    G:\\...\\venv_xpu\\Scripts\\python.exe research/tools/check_policy_compile.py \\
        --out h:\\...\\.venv_xpu\\runs\\compile_consistency.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_policy_lora import (  # noqa: E402
    K_STEPS, PolicyDataset, build_index, load_model, make_collate,
)


def run_one(inner, head, inputs, device):
    kw = {}
    for k in ("input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
              "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
        v = inputs.get(k, None)
        if v is not None:
            kw[k] = v.to(device)
    out = inner(**kw, use_cache=False)
    h = out.last_hidden_state
    idx = kw["attention_mask"].sum(-1) - 1
    pooled = h[torch.arange(h.shape[0], device=h.device), idx]
    pred = head(pooled.float()).view(-1, K_STEPS, 2)
    return pooled.float().cpu(), pred.cpu()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"G:\vrc_recording\vrc_model\weights\GameScaling-0.8B-S3")
    ap.add_argument("--run", default=r"h:\AI\neko-music\vrc\pc-vr\n.e.k.o_plugin_vrc_body\.venv_xpu\runs\m0_s3_448\best")
    ap.add_argument("--dataset", default=r"G:\vrc_recording\vrc_model\dataset\m0_policy_448")
    ap.add_argument("--size", nargs=2, type=int, default=[448, 256])
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    device = "xpu" if torch.xpu.is_available() else "cpu"
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = int(a.size[0]) * int(a.size[1])
    samples = [s for s in build_index(Path(a.dataset), None) if s["split"] == "val"][:a.samples]
    ds = PolicyDataset(samples, processor, min_px, max_px)
    collate = make_collate(processor, min_px, max_px)

    model, head, _ = load_model(a.model, 128, device)
    model.load_adapter(str(Path(a.run) / "adapter"), adapter_name="trained", is_trainable=False)
    model.set_adapter("trained")
    head.load_state_dict(torch.load(Path(a.run) / "head.pt", map_location=device))
    model.eval(); head.eval()

    batches = [collate([ds[i]]) for i in range(len(ds))]
    inner = model.base_model.model.model
    eager = []
    with torch.no_grad():
        for b in batches:
            eager.append(run_one(inner, head, b["inputs"], device))

    print("[compile] …", flush=True)
    model.base_model.model.model = torch.compile(inner)
    compiled = []
    with torch.no_grad():
        for b in batches:
            compiled.append(run_one(model.base_model.model.model, head, b["inputs"], device))

    hid_cos, pred_cos, pred_max, pred_mse = [], [], [], []
    for (he, pe), (hc, pc) in zip(eager, compiled):
        hid_cos.append(float(F.cosine_similarity(he.flatten(), hc.flatten(), dim=0)))
        pred_cos.append(float(F.cosine_similarity(pe.flatten(), pc.flatten(), dim=0)))
        pred_max.append(float((pe - pc).abs().max()))
        pred_mse.append(float(((pe - pc) ** 2).mean()))
    res = dict(
        samples=len(batches), device=device,
        input_ids=list(batches[0]["inputs"]["input_ids"].shape),
        hidden_cos=dict(min=min(hid_cos), mean=sum(hid_cos) / len(hid_cos),
                        p05=sorted(hid_cos)[max(0, len(hid_cos) // 20)]),
        pred_cos=dict(min=min(pred_cos), mean=sum(pred_cos) / len(pred_cos)),
        pred_absmax=dict(max=max(pred_max), mean=sum(pred_max) / len(pred_max)),
        pred_mse=dict(max=max(pred_mse), mean=sum(pred_mse) / len(pred_mse)),
        gate_hidden_cos_0_999=min(hid_cos) >= 0.999,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
