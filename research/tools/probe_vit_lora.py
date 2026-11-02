# -*- coding: utf-8 -*-
"""验证 ViT LoRA 注入是否真的生效（不训练，只装配 + 反传）。

检查四件事：
1. 视觉塔可训练参数是否 > 0（注入前是 0.00%）；
2. 反传后视觉塔参数是否拿到非零梯度；
3. 前向是否仍然跑得通、输出形状是否还是 (B, 12, 2)；
4. 新增的可训练参数量有多大（决定显存与吞吐代价）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoProcessor
    from train_policy_lora import (PolicyDataset, build_index, forward_batch,
                                   load_model, make_collate, masked_mse)

    device = "xpu" if torch.xpu.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = 448 * 256

    samples = [s for s in build_index(Path(a.dataset), None) if s["split"] == "train"][:a.batch]
    ds = PolicyDataset(samples, processor, min_px, max_px)
    batch = make_collate(processor, min_px, max_px)([ds[i] for i in range(len(samples))])

    for blocks in (0, a.blocks):
        print("=" * 60)
        print("blocks =", blocks)
        print("=" * 60)
        model, head, hidden = load_model(a.model, 128, device, vit_lora_blocks=blocks)
        vis_tr = vis_total = 0
        for name, p in model.named_parameters():
            if "visual" in name.lower():
                vis_total += 1
                if p.requires_grad:
                    vis_tr += 1
        n_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("  视觉塔参数 %d 个，其中可训练 %d 个" % (vis_total, vis_tr))
        print("  全模型可训练 %d (%.2fM)" % (n_all, n_all / 1e6))

        model.train()
        pred = forward_batch(model, head, batch["inputs"], device)
        loss = masked_mse(pred, batch["act"].to(device), batch["mask"].to(device))
        loss.backward()
        print("  前向/反传 ok: pred=%s loss=%.4f" % (tuple(pred.shape), loss.item()))

        vis_grad_nz = 0
        vis_grad_n = 0
        for name, p in model.named_parameters():
            if "visual" in name.lower() and p.grad is not None:
                vis_grad_n += 1
                if bool((p.grad != 0).any()):
                    vis_grad_nz += 1
        print("  视觉塔有梯度 %d 个，其中非零 %d 个" % (vis_grad_n, vis_grad_nz))
        if device == "xpu":
            print("  显存峰值 %.2f GiB" % (torch.xpu.max_memory_allocated() / 2 ** 30))
        del model, head
        if device == "xpu":
            torch.xpu.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
