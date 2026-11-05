# -*- coding: utf-8 -*-
r"""1188/1241 个「ATen 原样调用」到底是什么 —— 逐个归类，别猜。

已知（2026-11-03）：compile 后 3715 个 kernel，其中 1188 个是「ATen 原样调用」
（没被 Inductor 接管）。按类别占比 ATen 其它 33.9%、库 GEMM 26.1%、Triton 融合 24.3%。

已排除的错误猜测（都实测过）：
  * `n_spills` 融合阈值：放宽到 100000，kernel 数**一个没变**（8995），只是慢 2.4%
    ⇒ 那段代码根本没走到，问题不在 spill gate
  * `aggressive_fusion`：0.2% 差异，噪声内
  * GatedDeltaNet 参考实现：只占 GPU 5.6%

本脚本把前向里所有**非 Triton** 的 kernel 按名字精确统计，输出：
  1. 每个算子名的个数 / GPU 时间 / 是否 aten::；
  2. 按「aten:: 前缀族」聚合（如 aten::mm、aten::slice、aten::copy_）；
  3. 每条同时给出「若是融合候选，能省多少」的量级判断。
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
    ap.add_argument("--sessions", nargs="*", default=["20261031_020757"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader
    import train_policy_lora as T

    dev = "xpu"
    proc = AutoProcessor.from_pretrained("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    px = 448 * 256
    samples = [s for s in T.build_index(Path(a.dataset), None) if s["split"] == "train"][: a.batch * 3]
    ds = T.PolicyDataset(samples, proc, px, px)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0,
                    collate_fn=T.make_collate(proc, px, px))
    b = next(iter(dl))

    act_dim = int(np.asarray(samples[0]["act"]).shape[-1])
    model, head, hidden = T.load_model("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3",
                                       128, dev, act_dim=act_dim)
    model.eval()
    inner = model.base_model.model
    inner.model = torch.compile(inner.model)
    kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b["inputs"].items()}

    with torch.no_grad():
        for _ in range(3):
            inner(**kw, use_cache=False)
        torch.xpu.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
            inner(**kw, use_cache=False)
            torch.xpu.synchronize()

    rows = []
    for e in prof.key_averages():
        dt = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if dt <= 0:
            continue
        rows.append((str(e.key), int(e.count), dt / 1000.0))
    rows.sort(key=lambda x: -x[2])
    tot_ms = sum(r[2] for r in rows)
    tot_n = sum(r[1] for r in rows)

    # 按 aten:: 前缀族聚合
    fam = defaultdict(lambda: [0, 0.0])
    for name, n, ms in rows:
        if name.startswith("aten::"):
            key = name
        elif "triton_" in name.lower():
            key = "triton_* (已融合)"
        elif "gemm" in name.lower() or "onednn" in name.lower():
            key = "库 GEMM (oneDNN/MKL)"
        else:
            key = "xpU 内置 kernel"
        fam[key][0] += n
        fam[key][1] += ms

    print()
    print("总 kernel %d，总 GPU %.1f ms" % (tot_n, tot_ms))
    print()
    print("  %-34s %7s %11s %8s" % ("算子族", "个数", "GPU(ms)", "占比"))
    for k, (n, ms) in sorted(fam.items(), key=lambda x: -x[1][1])[: a.top]:
        print("  %-34s %7d %11.1f %7.1f%%" % (k[:34], n, ms, 100.0 * ms / max(tot_ms, 1e-9)))

    print()
    print("  --- 全部 aten:: 算子（按 GPU 时间排）---")
    print("  %-40s %7s %11s %8s" % ("算子", "个数", "GPU(ms)", "占比"))
    aten = [(nm, n, ms) for nm, n, ms in rows if nm.startswith("aten::")]
    for nm, n, ms in aten[: a.top]:
        print("  %-40s %7d %11.1f %7.1f%%" % (nm[:40], n, ms, 100.0 * ms / max(tot_ms, 1e-9)))

    print()
    aten_ms = sum(ms for _, _, ms in aten)
    print("  aten:: 合计 %.1f ms（%.1f%%），%d 个"
          % (aten_ms, 100.0 * aten_ms / max(tot_ms, 1e-9), sum(n for _, n, _ in aten)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
