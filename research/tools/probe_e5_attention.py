# -*- coding: utf-8 -*-
"""E5：末位 token 对视觉 token 的注意力质量 —— 判决「读出通路」假设。

**要裁决的分歧**（2026-11-02）：
  * Lead 的判断：视觉没被训练（ViT 冻结、可训练参数 0%），所以要解冻 ViT；
  * policy-designer 的判断：信息在视觉特征里（线性探针 -56% 证明），但
    forward_batch 只取**末位 token** 的 hidden（train_policy_lora.py:255 起），
    112 个视觉 token 的信息必须全挤进这一个位置才能到输出头 —— 瓶颈在通路不在表示。

**测量**：加载训好的模型，跑一批真实样本，取最后一层的注意力，
统计「末位 token 作为 query、对 112 个视觉 token 作为 key」的注意力质量占比。

判据（policy-designer 给的）：
  * < 0.05  => LM 几乎不看视觉  -> 直接做中间层特征注入
  * > 0.2 但输出不依赖图像      -> 视觉被读了但被下游忽略 -> 才轮到解冻 ViT

**注意**：Qwen3.5 是混合线性注意力（多数层是 GatedDeltaNet，没有标准 attn 矩阵）。
所以本脚本只统计**确实有注意力权重的全注意力层**，并报告是哪几层。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--run", default=".venv_xpu/runs/m0_s3_448/best")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoProcessor
    from train_policy_lora import (PolicyDataset, build_index, load_model,
                                   make_collate)

    device = "xpu" if torch.xpu.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = 448 * 256

    samples = [s for s in build_index(Path(a.dataset), None) if s["split"] == "val"][:a.n]
    ds = PolicyDataset(samples, processor, min_px, max_px)
    batch = make_collate(processor, min_px, max_px)([ds[i] for i in range(len(samples))])

    model, head, hidden = load_model(a.model, 128, device)
    model.load_adapter(str(Path(a.run) / "adapter"), adapter_name="trained", is_trainable=False)
    model.set_adapter("trained")
    head.load_state_dict(torch.load(Path(a.run) / "head.pt", map_location=device))
    model.eval()

    inner = model.base_model.model.model
    print("[model] inner=%s" % type(inner).__name__, flush=True)

    # 找出所有「确实有 attn 权重」的层
    attn_layers = []
    for i, layer in enumerate(getattr(inner, "layers", [])):
        att = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if att is not None and not hasattr(att, "chunk_gated_delta_rule"):
            attn_layers.append((i, att))
    print("[attn] 全注意力层: %s" % [i for i, _ in attn_layers], flush=True)

    captured = {}

    def make_hook(idx):
        def hook(module, args, kwargs, output):
            # output 可能是 (attn_out, weights, past) 或 dict
            w = None
            if isinstance(output, tuple):
                for x in output:
                    if torch.is_tensor(x) and x.dim() == 4:
                        w = x
                        break
            captured[idx] = w
        return hook

    handles = []
    for idx, att in attn_layers:
        try:
            handles.append(att.register_forward_hook(make_hook(idx), with_kwargs=True))
        except TypeError:
            handles.append(att.register_forward_hook(make_hook(idx)))

    kw = {k: v.to(device) for k, v in batch["inputs"].items()
          if k in ("input_ids", "attention_mask", "pixel_values",
                   "pixel_values_videos", "video_grid_thw", "mm_token_type_ids")}
    with torch.no_grad():
        inner(**kw, use_cache=False, output_attentions=True)

    for h in handles:
        h.remove()

    # 视觉 token 的位置：从 mm_token_type_ids 取（视觉标 1）
    mtt = kw.get("mm_token_type_ids")
    mask = kw["attention_mask"][0]
    valid = mask.to(torch.bool)
    if mtt is not None:
        vis_pos = (mtt[0].to(torch.bool) & valid).nonzero().squeeze(-1)
    else:
        vis_pos = torch.tensor([], dtype=torch.long)
    last = int(mask.sum().item()) - 1
    print("[tokens] 有效 %d，视觉 %d 个，末位 idx=%d"
          % (int(mask.sum()), len(vis_pos), last), flush=True)

    out = {"attn_layers": [i for i, _ in attn_layers], "last_idx": last,
           "n_visual": int(len(vis_pos)), "n_valid": int(mask.sum()), "per_layer": {}}

    if not captured or vis_pos.numel() == 0:
        print("[warn] 没抓到注意力权重（混合线性注意力下这是预期内的）")
        out["note"] = "没抓到标准 attn 权重"
    else:
        for idx, w in captured.items():
            if w is None:
                continue
            # w: (B, H, L, L)；取 batch 0，对 heads 求平均
            a0 = w[0].float().mean(0)          # (L, L)
            row = a0[last]                     # 末位 token 的注意力分布 (L,)
            mass_vis = float(row[vis_pos].sum().item())
            total = float(row.sum().item())
            frac = mass_vis / max(total, 1e-12)
            # 均匀分布下「112 个视觉 token 应得」的比例（视觉占有效 token 的比例）
            base = float(len(vis_pos)) / max(int(mask.sum()), 1)
            out["per_layer"][str(idx)] = dict(
                vis_mass=frac, uniform_baseline=base, ratio=frac / max(base, 1e-12))
            print("  layer %-3d 视觉注意力质量 %.4f（均匀基线 %.4f，比值 %.2fx）"
                  % (idx, frac, base, frac / max(base, 1e-12)), flush=True)

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "e5_attention.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (o / "e5_attention.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
