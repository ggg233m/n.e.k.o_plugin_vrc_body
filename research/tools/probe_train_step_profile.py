# -*- coding: utf-8 -*-
r"""训练一步到底花在哪 —— 用 profiler 给出确定的分解，别再靠 GPU 计数器猜。

Windows 的 \GPU Engine 计数器不可靠（本仓库已记：跨引擎求和是错的、实例数在变），
所以这里直接量三件事：
  1. 每步 wall；
  2. 每步 **GPU 忙碌时间**（profiler 的 device time 之和）；
  3. 每步 **kernel 个数** 与 CPU 自耗时。
利用率 = GPU 忙碌 / wall。若它只有 ~30-50% 而 kernel 数上万，说明瓶颈是逐 kernel 开销，
不是算力。再把 kernel 按名字归类（gemm / 卷积类 / 逐元素），就能定位是谁在拖。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
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


def bucket(name: str) -> str:
    n = name.lower()
    if "gemm" in n or "mm" == n or "aten::mm" in n or "matmul" in n:
        return "gemm"
    if "conv" in n or "causal_conv" in n:
        return "conv"
    if "copy" in n:
        return "copy(派发型)"
    if "elementwise" in n or "aten::add" in n or "aten::mul" in n or "aten::div" in n:
        return "elementwise"
    if "bmm" in n or "batch" in n:
        return "bmm"
    if n.startswith("aten::"):
        return "aten 其它"
    return "其它 kernel"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--sessions", nargs="*", default=["20261031_020757"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--skip-profile", action="store_true",
                    help="只测 wall，不跑 torch.profiler（profiler 本身会拖慢 wall）")
    ap.add_argument("--agg-fusion", action="store_true",
                    help="Inductor aggressive_fusion + max_fusion_size=256（预填充口径实测 -8%%）")
    ap.add_argument("--spill-threshold", type=int, default=None,
                    help="覆盖 Inductor 的 n_spills 融合拒绝阈值（默认 8）。实测 XPU 上 "
                         "n_spills 最大 46976（疑似按字节上报，对应 PR #197061）")
    a = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoProcessor
    import train_policy_lora as T

    dev = "xpu" if torch.xpu.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(T.DEFAULT_MODEL if hasattr(T, "DEFAULT_MODEL") else
                                         "G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    px = 448 * 256
    samples = T.build_index(Path(a.dataset), a.sessions)
    # 预热 3 + 测量 steps + 1 次 profile，按 batch 算条数再留余量
    need = a.batch * (a.steps + 6)
    samples = [s for s in samples if s["split"] == "train"][:need]
    if len(samples) < need:
        print("[warn] 可用训练样本只有 %d 条（需要 %d），结果可能受冷启动影响"
              % (len(samples), need), flush=True)
    ds = T.PolicyDataset(samples, proc, px, px, training=True)
    collate = T.make_collate(proc, px, px)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0, collate_fn=collate)

    act_dim = int(np.asarray(samples[0]["act"]).shape[-1])
    model, head, hidden = T.load_model("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3",
                                       128, dev, act_dim=act_dim)
    model.train()
    if a.spill_threshold is not None:
        # 运行时改阈值，不动磁盘源码（可回退、不影响其它进程）。
        # 依据：codegen/triton.py:8331 `if launchers[0].n_spills > n_spills_threshold`
        # 直接把 ms 设为 inf（连基准都不跑）⇒ 融合被拒。
        import torch._inductor.codegen.triton as _tk
        _cls = None
        for _n in dir(_tk):
            _o = getattr(_tk, _n)
            if isinstance(_o, type) and hasattr(_o, 'benchmark_fused_nodes'):
                _cls = _o
                break
        if _cls is None:
            print('[inductor] 找不到 benchmark_fused_nodes 所在类，跳过')
        else:
            _orig = _cls.benchmark_fused_nodes
            def _patched(self, nodes, n_spills_threshold=8, _o=_orig, _t=a.spill_threshold):
                return _o(self, nodes, n_spills_threshold=_t)
            _cls.benchmark_fused_nodes = _patched
            print("[inductor] n_spills 融合阈值 8 -> %d (via %s)" % (a.spill_threshold, _cls.__name__), flush=True)

    if a.agg_fusion:
        import torch._inductor.config as ic
        ic.aggressive_fusion = True
        ic.max_fusion_size = 256
        print("[inductor] aggressive_fusion=True max_fusion_size=256", flush=True)
    if a.compile:
        inner = model.base_model.model.model
        model.base_model.model.model = torch.compile(inner, dynamic=False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] +
                            list(head.parameters()), lr=1e-4)

    def one_step(batch):
        r = T.forward_batch(model, head, batch["inputs"], dev)
        pred = r[0] if isinstance(r, tuple) else r
        # act/mask 仍在 CPU（collate 不搬），必须显式上设备，否则 masked_mse 报跨设备
        act = batch["act"].to(dev)
        msk = batch["mask"].to(dev)
        loss = T.masked_mse(pred, act, msk)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return float(loss.detach())

    it = iter(dl)
    print("[warm] 预热…", flush=True)
    for _ in range(3):
        one_step(next(it))
    if dev == "xpu":
        torch.xpu.synchronize()

    walls = []
    for _ in range(a.steps):
        b = next(it)
        if dev == "xpu":
            torch.xpu.synchronize()
        t0 = time.perf_counter()
        one_step(b)
        if dev == "xpu":
            torch.xpu.synchronize()
        walls.append((time.perf_counter() - t0) * 1000)

    if a.skip_profile:
        ws = sorted(walls)
        print()
        print("  [wall-only] n=%d  p50=%.1f ms  min=%.1f ms  max=%.1f ms"
              % (len(ws), float(np.percentile(ws, 50)), ws[0], ws[-1]))
        return 0

    b = next(it)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.XPU] if dev == "xpu" else [])
    with profile(activities=acts) as prof:
        one_step(b)
        if dev == "xpu":
            torch.xpu.synchronize()

    per = defaultdict(lambda: [0, 0.0])   # bucket -> [count, device_ms]
    gpu_us = 0.0
    cpu_us = 0.0
    total_n = 0
    for e in prof.key_averages():
        dt = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        ct = float(getattr(e, "self_cpu_time_total", 0.0) or 0.0)
        cpu_us += ct
        if dt <= 0:
            continue
        k = bucket(str(e.key))
        per[k][0] += int(e.count)
        per[k][1] += dt / 1000.0
        gpu_us += dt
        total_n += int(e.count)

    wall = float(np.median(walls))
    print()
    print("  每步 wall（中位）      %.1f ms" % wall)
    print("  每步 GPU 忙碌          %.1f ms" % (gpu_us / 1000.0))
    print("  ⇒ GPU 利用率           %.1f%%" % (100.0 * gpu_us / 1000.0 / wall))
    print("  每步 kernel 个数       %d" % total_n)
    print("  每步 CPU 自耗时合计    %.1f ms" % (cpu_us / 1000.0))
    print()
    print("  %-16s %10s %12s %12s" % ("类别", "kernel数", "GPU忙(ms)", "占GPU%"))
    for k, (n, ms) in sorted(per.items(), key=lambda x: -x[1][1]):
        print("  %-16s %10d %12.1f %11.1f%%" % (k, n, ms, 100.0 * ms / (gpu_us / 1000.0)))
    print()
    fl = 2.0 * 0.896e9 * a.batch * 128 * 3   # fwd+bwd 粗估
    print("  粗估 FLOP/step %.3f TFLOP -> 实测算力 %.2f TFLOPS（%%峰值 %.1f）"
          % (fl / 1e12, fl / (wall / 1000.0) / 1e12, 100.0 * fl / (wall / 1000.0) / 40e12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
