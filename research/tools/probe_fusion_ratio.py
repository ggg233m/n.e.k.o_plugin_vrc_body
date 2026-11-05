# -*- coding: utf-8 -*-
r"""Triton 现在到底融合了多少 —— 按 kernel 来源分类，不靠估算。

三个数：
  1. eager 的 ATen 算子数（融合前的原子操作数）；
  2. compile 后实际启动的 kernel 数；
  3. 其中哪些是 Inductor/Triton **生成**的融合 kernel（名字含 triton_），
     哪些还是原样调库（aten::* / gemm_kernel / oneDNN）。

融合率 = 1 - compiled_kernels / eager_ops；
其中「真融合」= triton_* kernel 覆盖了多少原子操作。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_REPO / ".tc_cache" / "inductor"))
_ONEAPI = r"D:\Program Files (x86)\Intel\oneAPI\2026.0\bin"
if os.path.isdir(_ONEAPI):
    os.environ["PATH"] = _ONEAPI + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("CC", "icpx")
    os.environ.setdefault("CXX", "icpx")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def classify(key: str) -> str:
    k = str(key)
    kl = k.lower()
    if "triton_" in kl:
        return "Triton 融合 kernel"
    if kl.startswith("aten::"):
        return "ATen 原样调用"
    if "gemm" in kl or "onednn" in kl or "cblas" in kl:
        return "库 GEMM"
    if "kernel" in kl:
        return "xpU 内置 kernel"
    return "其它"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3near")
    ap.add_argument("--sessions", nargs="*", default=["20261031_020757"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--mode", choices=["fwd", "train"], default="fwd")
    a = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader
    import train_policy_lora as T

    dev = "xpu"
    proc = AutoProcessor.from_pretrained("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    px = 448 * 256
    samples = T.build_index(Path(a.dataset), a.sessions)
    samples = [s for s in samples if s["split"] == "train"][: a.batch * 6]
    ds = T.PolicyDataset(samples, proc, px, px, training=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0,
                    collate_fn=T.make_collate(proc, px, px))

    act_dim = int(np.asarray(samples[0]["act"]).shape[-1])
    model, head, hidden = T.load_model("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3",
                                       128, dev, act_dim=act_dim)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] +
                            list(head.parameters()), lr=1e-4)

    def run(batch, train: bool):
        r = T.forward_batch(model, head, batch["inputs"], dev)
        pred = r[0] if isinstance(r, tuple) else r
        loss = T.masked_mse(pred, batch["act"].to(dev), batch["mask"].to(dev))
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return loss

    it = iter(dl)
    b = next(it)

    # --- 1. eager：数 ATen 算子 ---
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as p0:
        run(b, train=(a.mode == "train"))
        torch.xpu.synchronize()
    eager_ops = sum(int(e.count) for e in p0.key_averages()
                    if float(getattr(e, "self_device_time_total", 0.0) or 0.0) > 0)
    if eager_ops == 0:
        # CPU-only profile 拿不到 device time；退回按 kernel 名计数
        eager_ops = sum(int(e.count) for e in p0.key_averages()
                        if classify(str(e.key)) != "其它")
    print("[eager] ops=%d" % eager_ops, flush=True)

    # --- 2. compile ---
    inner = model.base_model.model
    inner.model = torch.compile(inner.model)
    for i in range(3):
        run(b, train=(a.mode == "train"))
    torch.xpu.synchronize()
    print("[compile] 预热完成", flush=True)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as p1:
        run(b, train=(a.mode == "train"))
        torch.xpu.synchronize()

    buckets = {}
    for e in p1.key_averages():
        dt = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if dt <= 0:
            continue
        k = classify(str(e.key))
        n, ms = buckets.get(k, (0, 0.0))
        buckets[k] = (n + int(e.count), ms + dt / 1000.0)

    comp_ops = sum(v[0] for v in buckets.values())
    print()
    print("  mode              %s" % a.mode)
    print("  eager ATen 算子数 %d" % eager_ops)
    print("  compile 后 kernel %d" % comp_ops)
    print("  >>> 融合率        %.1f%%  (1 - %d/%d)"
          % (100.0 * (1 - comp_ops / max(eager_ops, 1)), comp_ops, eager_ops))
    print()
    print("  %-22s %8s %12s %10s" % ("来源", "个数", "GPU(ms)", "占比"))
    tot_ms = sum(v[1] for v in buckets.values())
    for k, (n, ms) in sorted(buckets.items(), key=lambda x: -x[1][1]):
        print("  %-22s %8d %12.1f %9.1f%%" % (k, n, ms, 100.0 * ms / max(tot_ms, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
