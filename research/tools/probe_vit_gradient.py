# -*- coding: utf-8 -*-
"""查 ViT 到底有没有可训练参数、视觉 token 的梯度能不能到任何权重。

背景（2026-11-02 视觉消融）：固定 wp/body/文本，把图像换成全黑只让动作块偏移 1.7%，
换成别的场景偏移 2.0%。模型几乎不看图。先排除最可能的解释：
**ViT 全冻结，且 LoRA 一个都没挂在视觉塔上** —— 那视觉特征对这个任务就是常数。

本脚本做四件事：
1. 列出视觉塔与 LM 各自的模块名，标出 TARGET_MODULES 里哪些名字出现在哪边；
2. 统计 requires_grad 的参数分布（视觉塔 / LM / 其他）；
3. 真跑一次前向 + 反传，检查视觉塔参数的 .grad 是否非空（若不是 None，说明
   还有别的梯度路径；若全是 None，确认视觉分支完全没被训练）；
4. 检查视觉输出是否随输入改变（同一张图 vs 全黑图，pooler 输出的差异）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


TARGET_MODULES = [
    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--run", default=".venv_xpu/runs/m0_s3_448/best")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoProcessor
    from train_policy_lora import load_model, PolicyDataset, build_index, make_collate

    device = "xpu" if torch.xpu.is_available() else "cpu"
    model, head, hidden = load_model(a.model, 128, device)

    # ---------- 1. 结构：视觉塔有哪些模块名，哪些命中 TARGET_MODULES ----------
    print("=" * 70)
    print("1. 模块结构")
    print("=" * 70)
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if "visual" in name.lower() and cls in ("Linear", "Qwen3_5VisionBlock", "Qwen3_5VisionAttention"):
            print("  %-60s %s" % (name[:60], cls))
    print()

    # 视觉塔里的 Linear 名字集合
    vis_linear_names = set()
    lm_linear_names = set()
    for name, mod in model.named_modules():
        if type(mod).__name__ == "Linear":
            leaf = name.rsplit(".", 1)[-1]
            if "visual" in name.lower():
                vis_linear_names.add(leaf)
            else:
                lm_linear_names.add(leaf)
    print("  视觉塔 Linear 叶名: %s" % sorted(vis_linear_names))
    print("  LM     Linear 叶名: %s" % sorted(lm_linear_names))
    print("  TARGET_MODULES 命中视觉塔: %s"
          % sorted(set(TARGET_MODULES) & vis_linear_names))
    print("  TARGET_MODULES 命中 LM:     %s"
          % sorted(set(TARGET_MODULES) & lm_linear_names))
    print()

    # ---------- 2. LoRA 到底挂了哪些模块 ----------
    print("=" * 70)
    print("2. 可训练参数分布")
    print("=" * 70)
    tot = 0
    buckets = {"visual": 0, "lm": 0, "other": 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        tot += n
        if "visual" in name.lower():
            buckets["visual"] += n
        elif "lora" in name.lower():
            buckets["lm"] += n
        else:
            buckets["other"] += n
    print("  可训练合计 %d" % tot)
    for k, v in buckets.items():
        print("    %-8s %d (%.2f%%)" % (k, v, 100.0 * v / max(tot, 1)))
    head_n = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print("  ActionHead 可训练 %d" % head_n)
    print()

    # ---------- 3. 视觉塔有没有非 None 的梯度 ----------
    print("=" * 70)
    print("3. 反传检查")
    print("=" * 70)
    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = 448 * 256
    samples = [s for s in build_index(Path(a.dataset), None) if s["split"] == "val"][:2]
    ds = PolicyDataset(samples, processor, min_px, max_px)
    collate = make_collate(processor, min_px, max_px)
    batch = collate([ds[0], ds[1]])

    model.train()
    from train_policy_lora import forward_batch, masked_mse
    pred = forward_batch(model, head, batch["inputs"], device)
    loss = masked_mse(pred, batch["act"].to(device), batch["mask"].to(device))
    loss.backward()
    print("  loss=%.4f" % loss.item())

    vis_with_grad, vis_total, vis_nonzero = 0, 0, 0
    for name, p in model.named_parameters():
        if "visual" in name.lower():
            vis_total += 1
            if p.grad is not None:
                vis_with_grad += 1
                if bool((p.grad != 0).any()):
                    vis_nonzero += 1
    print("  视觉塔参数 %d 个：有 .grad 的 %d 个，其中非零 %d 个"
          % (vis_total, vis_with_grad, vis_nonzero))
    print("  → 视觉塔全部 requires_grad=False 时，这里应当 0/0/0（没有梯度也不需要）")
    print()

    # ---------- 4. 视觉特征是否随输入改变 ----------
    print("=" * 70)
    print("4. 视觉塔输出对输入的敏感度（这才是根因所在）")
    print("=" * 70)
    model.eval()
    import torch as T
    base_mod = model.base_model.model  # PeftModel -> LoraModel -> ForConditionalGeneration
    inner = base_mod.model
    visual = inner.visual

    def vit_feat(video_np, text):
        out = processor(text=[text], videos=[video_np], return_tensors="pt",
                        padding=True, min_pixels=min_px, max_pixels=max_px,
                        cap_pixels_per_frame=False, do_sample_frames=False)
        pv = out["pixel_values_videos"].to(device)
        grid = out["video_grid_thw"].to(device)
        with T.no_grad():
            feats = visual(pv, grid_thw=grid)
        return feats.pooler_output.float().cpu().numpy()

    s = samples[0]
    from PIL import Image
    cur = np.asarray(Image.open(s["img"]).convert("RGB"))
    prev = np.asarray(Image.open(s["prev_img"]).convert("RGB"))
    video_real = np.stack([prev, cur])
    video_black = np.zeros_like(video_real)

    item = ds[0]
    text = item["text"]
    f_real = vit_feat(video_real, text)
    f_black = vit_feat(video_black, text)
    d = f_real - f_black
    cos = float((f_real.ravel() @ f_black.ravel())
                / (np.linalg.norm(f_real) * np.linalg.norm(f_black) + 1e-12))
    print("  视觉特征维度 %s" % (f_real.shape,))
    print("  真实图 vs 全黑图：|diff| mean=%.4f  max=%.4f  cos=%.4f"
          % (np.abs(d).mean(), np.abs(d).max(), cos))
    print("  特征绝对值 mean=%.4f" % np.abs(f_real).mean())
    print()
    print("  ⚠️ 若 cos 接近 1 且 |diff| 远小于特征绝对值，说明视觉塔几乎不区分输入。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
