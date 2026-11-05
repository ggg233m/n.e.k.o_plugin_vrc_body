# -*- coding: utf-8 -*-
r"""反向传播占 43% —— 那 1520 ms（eager, batch4）花在哪些算子上？

背景（task-16 实测）：
  纯 fwd 2237.8 ms / fwd+bwd 3758.1 ms ⇒ **反向 1520 ms，占整步 43%**。
  而 Lead 之前所有优化（编译融合 / spill 阈值 / batch）都是**前向视角**，
  从未单独看过反向。这个脚本补上这一块。

做法：用 autograd profiler 只包 **backward()**，按算子统计 device_time。
与「同样的算子在前向里花多少」对比，就能看出反向特有的开销。
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
    ap.add_argument("--top", type=int, default=25)
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
    model.train()
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    act = b["act"].to(dev)
    msk = b["mask"].to(dev)
    kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b["inputs"].items()}

    def fwd():
        r = T.forward_batch(model, head, kw, dev)
        p = r[0] if isinstance(r, tuple) else r
        return T.masked_mse(p, act, msk)

    # 预热
    for _ in range(2):
        loss = fwd(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.xpu.synchronize()

    # 1) 干净 wall 分段（不用 profiler）
    tw = []
    for _ in range(3):
        torch.xpu.synchronize(); t0 = __import__("time").perf_counter()
        loss = fwd()
        torch.xpu.synchronize(); t1 = __import__("time").perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.xpu.synchronize(); t2 = __import__("time").perf_counter()
        opt.step()
        torch.xpu.synchronize(); t3 = __import__("time").perf_counter()
        tw.append(((t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000))
    f_ms, b_ms, o_ms = (float(np.median([x[i] for x in tw])) for i in range(3))
    print()
    print("  [clean wall] fwd=%.1f ms  bwd=%.1f ms  opt=%.1f ms  total=%.1f ms"
          % (f_ms, b_ms, o_ms, f_ms + b_ms + o_ms))
    print("  ⇒ 反向占整步 %.1f%%" % (100.0 * b_ms / max(f_ms + b_ms + o_ms, 1e-9)))

    # 2) 只 profile backward
    loss = fwd()
    opt.zero_grad(set_to_none=True)
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as pb:
        loss.backward()
        torch.xpu.synchronize()
    rows = []
    gpu = 0.0
    for e in pb.key_averages():
        dt = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if dt <= 0:
            continue
        gpu += dt
        rows.append((str(e.key), int(e.count), dt / 1000.0))
    rows.sort(key=lambda x: -x[2])
    print()
    print("  [bwd profile] 事件 %d 个，GPU 合计 %.1f ms" % (sum(r[1] for r in rows), gpu / 1000.0))
    print()
    print("  %-46s %8s %11s %8s" % ("算子", "个数", "GPU(ms)", "占bwd%"))
    for nm, n, ms in rows[: a.top]:
        print("  %-46s %8d %11.1f %7.1f%%" % (nm[:46], n, ms, 100.0 * ms / max(gpu / 1000.0, 1e-9)))

    # 3) 归类
    bucket = defaultdict(lambda: [0, 0.0])
    for nm, n, ms in rows:
        l = nm.lower()
        if "mm" == l or "gemm" in l or "aten::mm" in l:
            k = "GEMM"
        elif "aten::copy_" in l or "copy" in l and "unrolled" in l:
            k = "copy"
        elif "aten::sum" in l or "reduce" in l:
            k = "reduce"
        elif "elementwise" in l or l.startswith("aten::add") or l.startswith("aten::mul") or l.startswith("aten::div"):
            k = "elementwise"
        elif "sigmoid" in l or "exp" in l or "tanh" in l or "silu" in l or "sqrt" in l:
            k = "激活/特殊函数"
        elif "sparse" in l or "scatter" in l or "index" in l:
            k = "scatter/index (LoRA/embedding 反向)"
        else:
            k = "其它"
        bucket[k][0] += n
        bucket[k][1] += ms
    print()
    print("  --- 反向按类别 ---")
    for k, (n, ms) in sorted(bucket.items(), key=lambda x: -x[1][1]):
        print("  %-34s %7d %10.1f ms %7.1f%%" % (k, n, ms, 100.0 * ms / max(gpu / 1000.0, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
