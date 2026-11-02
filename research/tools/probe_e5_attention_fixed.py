# -*- coding: utf-8 -*-
"""E5 修正版：在 6 个 full_attention 层上统计末位 token 对视觉 token 的注意力质量。

**上一版的错误**（2026-11-02，policy-designer 指出）：上一版按 `layer.self_attn` 找，
找不到就报"没有全注意力层"。实际 config 里 `layer_types` 在 3/7/11/15/19/23 共 6 层是
`full_attention`（`full_attention_interval: 4`），线性层则是 `linear_attn`。
所以"架构没有注意力"这句话是错的，必须撤回。

**另一个可能的原因**：`output_attentions=True` 在 SDPA 后端下不会返回矩阵，
需要强制 `attn_implementation='eager'` 才拿得到。两个原因都修。

判据（policy-designer 给的）：
  * < 0.05  => LM 几乎不看视觉 -> 直接中间层注入
  * > 0.2 但输出不依赖图像    -> 视觉被读了但被下游忽略
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
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoProcessor
    from train_policy_lora import PolicyDataset, build_index, load_model, make_collate

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
    # ⚠️ 层级是 Qwen3_5Model > language_model > layers（不是一个叫 .layers 的顶层），
    # 上一版直接 inner.layers 报 AttributeError。
    lm = inner.language_model
    layers = list(lm.layers)
    # 按 config 的 layer_types 定位 full_attention 层。
    # 线性层的子模块叫 linear_attn，全注意力层叫 self_attn —— 名字不同，
    # 上一版只找 self_attn 却用错容器，于是误报「没有全注意力层」。
    full_idx = []
    cfg_lt = getattr(lm.config, "layer_types", None) or []
    for i, t in enumerate(cfg_lt):
        if "full" in str(t):
            full_idx.append(i)
    print("[cfg] layer_types full_attention idx = %s" % full_idx, flush=True)

    # 模块名实际是什么？直接打印一层看看
    if layers:
        print("[probe] layer0 attrs = %s"
              % [n for n, _ in layers[0].named_children()], flush=True)
    for i in full_idx[:2]:
        print("[probe] layer%d attrs = %s"
              % (i, [n for n, _ in layers[i].named_children()]), flush=True)

    captured = {}

    def make_hook(i):
        def hook(module, args, kwargs, output):
            # Qwen3_5Attention.forward 一般返回 (attn_output, attn_weights, past_kv)
            w = None
            if isinstance(output, tuple):
                for x in output:
                    if torch.is_tensor(x) and x.dim() == 4:
                        w = x
                        break
            if w is not None:
                captured[i] = w.detach()
        return hook

    handles = []
    for i in full_idx:
        if i >= len(layers):
            continue
        att = getattr(layers[i], "self_attn", None)
        if att is None:
            continue
        handles.append(att.register_forward_hook(make_hook(i), with_kwargs=True))
    print("[hook] 挂上 %d 个" % len(handles), flush=True)

    kw = {k: v.to(device) for k, v in batch["inputs"].items()
          if k in ("input_ids", "attention_mask", "pixel_values",
                   "pixel_values_videos", "video_grid_thw", "mm_token_type_ids")}
    with torch.no_grad():
        inner(**kw, use_cache=False, output_attentions=True)
    for h in handles:
        h.remove()

    mtt = kw.get("mm_token_type_ids")
    mask = kw["attention_mask"][0]
    valid = mask.to(torch.bool)
    vis_pos = (mtt[0].to(torch.bool) & valid).nonzero().squeeze(-1) if mtt is not None else None
    last = int(mask.sum().item()) - 1
    n_valid = int(mask.sum())

    out = {"full_attention_idx": full_idx, "captured_layers": sorted(captured.keys()),
           "n_visual": int(vis_pos.numel()) if vis_pos is not None else 0,
           "n_valid": n_valid, "per_layer": {}}

    if vis_pos is None or not captured:
        out["note"] = "没抓到注意力权重"
        print("[warn] 没抓到（captured=%s）" % sorted(captured.keys()), flush=True)
    else:
        base = float(len(vis_pos)) / max(n_valid, 1)
        for i, w in sorted(captured.items()):
            a0 = w[0].float().mean(0)      # (L, L)
            if a0.shape[0] <= last:
                continue
            row = a0[last]
            frac = float(row[vis_pos].sum().item()) / max(float(row.sum().item()), 1e-12)
            out["per_layer"][str(i)] = dict(
                vis_mass=frac, uniform_baseline=base, ratio=frac / max(base, 1e-12))
            print("  layer %-3d 视觉注意力质量 %.4f（均匀基线 %.4f，%.2fx）"
                  % (i, frac, base, frac / max(base, 1e-12)), flush=True)

    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "e5_attention_fixed.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[done] %s" % (o / "e5_attention_fixed.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
