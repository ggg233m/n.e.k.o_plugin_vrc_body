# -*- coding: utf-8 -*-
"""量一下数据管线有多慢 —— 训练时 GPU 占用低的嫌疑源。

问题（2026-11-02，用户提出）：训练时显卡占用很低。

已知参照：方案 §11.3 在 A770 上测得 batch 4 / chunk 128 = 906 ms/step、4.41 样本/s。
但那次用的是随机 token（.tmp/lora_smoke_xpu.py），完全绕开数据加载。
现在实测 1174-1200 ms/step、3.4 样本/s —— 慢了约 25%。

本脚本把「每个 batch 的 CPU 侧耗时」拆开量：
  * 图像打开 + 解码（PIL，2 帧/样本）
  * processor（chat template + tokenize + 视觉预处理）
  * 合计（= GPU 每步要等多久）

若合计在 250-300 ms 量级，说明 GPU 有 1/4 的步时间在等 CPU，
根本原因是 PolicyDataset 的 num_workers=0。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pct(xs, p):
    return float(np.percentile(np.asarray(xs, np.float64), p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--sessions", nargs="*",
                    default=["20261031_020757", "20261031_023239",
                             "20261031_024251", "20261031_030127"])
    a = ap.parse_args(argv)

    from PIL import Image
    from transformers import AutoProcessor
    from train_policy_lora import PolicyDataset, build_index, make_collate

    processor = AutoProcessor.from_pretrained(a.model)
    min_px = max_px = 448 * 256
    samples = [s for s in build_index(Path(a.dataset), a.sessions) if s["split"] == "train"]
    print("[data] train=%d" % len(samples), flush=True)
    ds = PolicyDataset(samples, processor, min_px, max_px, training=True,
                       vision_dropout=0.2, seed=0)
    collate = make_collate(processor, min_px, max_px)

    t_img = []
    for i in range(min(40, len(ds))):
        s = samples[i]
        t = time.perf_counter()
        np.asarray(Image.open(s["img"]).convert("RGB"))
        np.asarray(Image.open(s["prev_img"]).convert("RGB"))
        t_img.append((time.perf_counter() - t) * 1000)
    print("  [image] 2 帧解码 p50=%.1f ms (每样本)" % pct(t_img, 50), flush=True)

    t_item = []
    for i in range(min(40, len(ds))):
        t = time.perf_counter()
        ds[i]
        t_item.append((time.perf_counter() - t) * 1000)
    print("  [item ] __getitem__ p50=%.1f ms (每样本)" % pct(t_item, 50), flush=True)

    items = [ds[i] for i in range(a.batch * 2)]
    t_col = []
    for k in range(a.iters):
        chunk = items[:a.batch]
        t = time.perf_counter()
        collate(chunk)
        t_col.append((time.perf_counter() - t) * 1000)
    print("  [coll ] collate(batch=%d) p50=%.1f ms" % (a.batch, pct(t_col, 50)), flush=True)

    t_all = []
    for k in range(a.iters):
        t = time.perf_counter()
        chunk = [ds[(k * a.batch + j) % len(ds)] for j in range(a.batch)]
        collate(chunk)
        t_all.append((time.perf_counter() - t) * 1000)
    per_step = pct(t_all, 50)
    print("  [total] 一个 batch 的数据准备 p50=%.1f ms" % per_step, flush=True)

    print()
    print("  每样本 __getitem__ %.1f ms x %d = %.0f ms/step（单进程）"
          % (pct(t_item, 50), a.batch, pct(t_item, 50) * a.batch), flush=True)
    print("  加上 collate 后合计 %.0f ms/step" % per_step, flush=True)
    print("  参照：方案 §11.3 纯算力 906 ms/step；当前实测 1174-1200 ms/step", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
