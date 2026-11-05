# -*- coding: utf-8 -*-
r"""1188 个「ATen 原样调用」里，GatedDeltaNet 占多少 GPU 时间？

背景（2026-11-03 实测）：一次前向 eager 9892 算子 → compile 后 3715 kernel，
其中 1188 个是 `ATen 原样调用`（没被 Inductor 接管）。若这些主要是 GatedDeltaNet
的参考实现落回，那「改上游」才值得；若它们只占 GPU 时间的一小部分，断图优化到此为止。

测法：用 torch.profiler 抓一次 compile 后的前向，按 kernel 名归类：
  * GatedDeltaNet 特征：`conv`/`delta`/`solve`/`triangular`/`as_strided`/一批小 `mm`
  * 并与「18 层 × 每层固定数量」的假设对照（层数能解释的部分）。
输出每一类占的 GPU 时间与算子数，最后给「值得改上游吗」的判据。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
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


GDN_HINTS = ("conv", "delta", "solve", "triangular", "stride", "expand", "cumsum",
             "softplus", "sigmoid", "gated")


def tag(key: str) -> str:
    k = str(key)
    kl = k.lower()
    if "triton_" in kl:
        return "Triton 融合"
    if "gemm" in kl or "onednn" in kl or "cblas" in kl:
        return "库 GEMM"
    if any(h in kl for h in GDN_HINTS):
        return "疑似 GatedDeltaNet 相关"
    if kl.startswith("aten::"):
        return "ATen 其它"
    return "其它"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3near")
    ap.add_argument("--sessions", nargs="*", default=["20261031_020757"])
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader
    import train_policy_lora as T

    torch._dynamo.config.capture_scalar_outputs = True
    dev = "xpu"
    proc = AutoProcessor.from_pretrained("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    px = 448 * 256
    samples = T.build_index(Path(a.dataset), a.sessions)
    samples = [s for s in samples if s["split"] == "train"][: a.batch * 6]
    ds = T.PolicyDataset(samples, proc, px, px, training=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0,
                    collate_fn=T.make_collate(proc, px, px))
    b = next(iter(dl))

    act_dim = int(np.asarray(samples[0]["act"]).shape[-1])
    model, head, hidden = T.load_model("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3",
                                       128, dev, act_dim=act_dim)
    model.eval()
    inner = model.base_model.model
    inner.model = torch.compile(inner.model)

    with torch.no_grad():
        for _ in range(3):
            T.forward_batch(model, head, b["inputs"], dev)
        torch.xpu.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
            T.forward_batch(model, head, b["inputs"], dev)
            torch.xpu.synchronize()

    buckets = defaultdict(lambda: [0, 0.0])
    rows = []
    for e in prof.key_averages():
        dt = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if dt <= 0:
            continue
        t = tag(str(e.key))
        buckets[t][0] += int(e.count)
        buckets[t][1] += dt / 1000.0
        rows.append((dt / 1000.0, int(e.count), str(e.key)[:70]))

    tot_n = sum(v[0] for v in buckets.values())
    tot_ms = sum(v[1] for v in buckets.values())
    print()
    print("  总 kernel %d，总 GPU %.1f ms" % (tot_n, tot_ms))
    print()
    print("  %-24s %8s %12s %10s" % ("类别", "kernel数", "GPU(ms)", "占比"))
    for k, (n, ms) in sorted(buckets.items(), key=lambda x: -x[1][1]):
        print("  %-24s %8d %12.1f %9.1f%%" % (k, n, ms, 100.0 * ms / max(tot_ms, 1e-9)))

    print()
    print("  --- GPU 时间最长的 15 个 kernel ---")
    rows.sort(reverse=True)
    for ms, n, name in rows[:15]:
        print("    %8.2f ms  n=%-5d %s" % (ms, n, name))

    gdn = buckets.get("疑似 GatedDeltaNet 相关", [0, 0.0])
    print()
    print("  判据：GatedDeltaNet 相关占 GPU 时间 %.1f%%（%d 个 kernel）"
          % (100.0 * gdn[1] / max(tot_ms, 1e-9), gdn[0]))
    print("  若 <5%%，改上游不值；若 >15%%，值得投入。" )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
