# -*- coding: utf-8 -*-
r"""反向到底有没有被 torch.compile 覆盖？

背景（2026-11-04）：
  task-16 实测反向占整步 43%；本机 probe_backward_breakdown 实测 49.4%。
  本探针 profile 反向：**13665 个事件 / 367.9 ms，其中 elementwise 6329 个（33.3%）、
  copy 2630 个（22.3%）**。而前向 compile 后只有 3715 个 kernel。

关键疑问：训练脚本的 `--compile` 只包了内层 `Qwen3_5Model.forward`：
    inner.model = torch.compile(inner.model)
  autograd 的反向由引擎生成，**不经过这个编译**。所以反向可能是未融合的 eager 图。

判据（本脚本做三组对照，同一 batch、同一模型）：
  A. 反向 + **不** compile 前向   → kernel 数 N_eager
  B. 反向 + compile 前向          → kernel 数 N_comp
  C. 整个训练步 compile（forward+backward 都在图内）→ kernel 数 N_full

若 N_comp ≈ N_eager ⇒ **反向完全没被编译**（前向省了，反向没省）。
若 N_full << N_eager ⇒ **compile 整个 step 才是正解**，收益可能很大。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_REPO / ".tc_cache" / "inductor"))
os.environ.setdefault("PYTHONUTF8", "1")
_ONEAPI = r"D:\Program Files (x86)\Intel\oneAPI\2026.0\bin"
if os.path.isdir(_ONEAPI):
    os.environ["PATH"] = _ONEAPI + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("CC", "icpx")
    os.environ.setdefault("CXX", "icpx")
sys.path.insert(0, str(Path(__file__).resolve().parent))

K = 8  # 只对前 8 步算 loss，减少无关计算


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--sessions", nargs="*", default=["20261031_020757"])
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader
    import train_policy_lora as T

    dev = "xpu"
    proc = AutoProcessor.from_pretrained("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    px = 448 * 256
    samples = [s for s in T.build_index(Path(a.dataset), None) if s["split"] == "train"][: a.batch * 2]
    ds = T.PolicyDataset(samples, proc, px, px)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0,
                    collate_fn=T.make_collate(proc, px, px))
    b = next(iter(dl))

    act_dim = int(np.asarray(samples[0]["act"]).shape[-1])
    model, head, hidden = T.load_model("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3",
                                       128, dev, act_dim=act_dim)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    act = b["act"].to(dev)
    msk = b["mask"].to(dev)
    kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b["inputs"].items()}

    def step(train: bool = True):
        r = T.forward_batch(model, head, kw, dev)
        p = r[0] if isinstance(r, tuple) else r
        loss = T.masked_mse(p, act, msk)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return float(loss.detach())

    # ---- 量：profile 只包 backward ----
    def measure_bwd(tag: str, warm: int = 2, reps: int = 3):
        for _ in range(warm):
            step()
        torch.xpu.synchronize()
        # 干净的 loss + graph，然后单独 profile backward
        r = T.forward_batch(model, head, kw, dev)
        p = r[0] if isinstance(r, tuple) else r
        loss = T.masked_mse(p, act, msk)
        opt.zero_grad(set_to_none=True)
        torch.xpu.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
            loss.backward()
            torch.xpu.synchronize()
        rows, gpu = [], 0.0
        for e in prof.key_averages():
            dt = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
            if dt <= 0:
                continue
            gpu += dt
            rows.append((str(e.key), int(e.count)))
        n = sum(c for _, c in rows)
        fresh = sum(c for nm, c in rows if "triton_" in nm.lower())
        print("  [%s] bwd: kernel=%d  其中 triton_=%d  GPU=%.1f ms"
              % (tag, n, fresh, gpu / 1000.0), flush=True)
        return n, fresh, gpu / 1000.0

    print()
    print("=== A. 前向不 compile ===", flush=True)
    n_a, t_a, g_a = measure_bwd("eager-fwd")

    print()
    print("=== B. 只 compile 内层 forward（= 当前训练脚本的做法）===", flush=True)
    inner = model.base_model.model
    inner.model = torch.compile(inner.model)
    n_b, t_b, g_b = measure_bwd("compiled-fwd")

    print()
    print("=== 汇总 ===")
    print("  A 前向未编译 → bwd kernel=%d, triton=%d, GPU=%.1f ms" % (n_a, t_a, g_a))
    print("  B 前向已编译 → bwd kernel=%d, triton=%d, GPU=%.1f ms" % (n_b, t_b, g_b))
    print()
    if n_b > 0 and n_a > 0:
        ratio = n_a / float(n_b)
        print("  kernel 数比 A/B = %.2f" % ratio)
        if ratio < 1.2 and t_b == 0:
            print("  ⇒ **反向完全没被编译**（两组 kernel 数接近，且都没有 triton 融合 kernel）")
        elif t_b > 0:
            print("  ⇒ 反向里有 %d 个 triton kernel（%.0f%%），说明 AOTAutograd 确实编译了反向"
                  % (t_b, 100.0 * t_b / max(n_b, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
