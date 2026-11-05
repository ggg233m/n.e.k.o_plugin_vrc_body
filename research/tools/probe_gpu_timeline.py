# -*- coding: utf-8 -*-
r"""训练每步那 40% 的等待到底在等什么 —— 不看 Windows GPU 计数器，只看 profile。

已知：wall 611 ms（compile），而 Windows `GPU Engine` 的 compute 采样在 1.8%~56.3% 之间乱跳，
不可信。要判「是 GPU 算得慢」还是「CPU 喂不上」，看两个比值：

  * GPU 忙 / wall        —— 越高说明 GPU 越满；
  * CPU 自耗时 / wall    —— 若 CPU 侧总时长明显超过 wall，说明 CPU 是并行的且在追。

另外按**时间线**看：profile 里每个 GPU kernel 的起止时间能给出真实的占用曲线，
不用再看那个计数器。这里输出 GPU 时间线的「忙碌片段覆盖率」。
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3near")
    ap.add_argument("--sessions", nargs="*", default=["20261031_020757"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--compile", action="store_true", default=True)
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
    samples = [s for s in samples if s["split"] == "train"][: a.batch * 8]
    ds = T.PolicyDataset(samples, proc, px, px, training=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0,
                    collate_fn=T.make_collate(proc, px, px))

    act_dim = int(np.asarray(samples[0]["act"]).shape[-1])
    model, head, hidden = T.load_model("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3",
                                       128, dev, act_dim=act_dim)
    model.train()
    inner = model.base_model.model
    inner.model = torch.compile(inner.model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] +
                            list(head.parameters()), lr=1e-4)

    def step(batch):
        r = T.forward_batch(model, head, batch["inputs"], dev)
        pred = r[0] if isinstance(r, tuple) else r
        loss = T.masked_mse(pred, batch["act"].to(dev), batch["mask"].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    it = iter(dl)
    for _ in range(4):
        step(next(it))
    torch.xpu.synchronize()

    # 采集 GPU 时间线
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        step(next(it))
        torch.xpu.synchronize()

    ev = [e for e in prof.events() if str(getattr(e, "device_type", "")).lower().endswith("xpu")
          and float(getattr(e, "device_time", 0) or 0) > 0]
    if not ev:
        print("[warn] 没拿到 XPU 事件"); return 1
    t0 = min(float(e.time_range.start) for e in ev)
    t1 = max(float(e.time_range.end) for e in ev)
    span_ms = (t1 - t0) / 1000.0
    busy_ms = sum(float(e.device_time) for e in ev) / 1000.0

    # 忙碌覆盖率：把区间离散化后统计有多少时间片内有 kernel 在跑
    grid = 200
    occ = np.zeros(grid, dtype=bool)
    for e in ev:
        s = int((float(e.time_range.start) - t0) / max(t1 - t0, 1e-9) * (grid - 1))
        en = int((float(e.time_range.end) - t0) / max(t1 - t0, 1e-9) * (grid - 1))
        occ[max(0, s):min(grid, en + 1)] = True

    print()
    print("  GPU 时间线跨度        %.1f ms" % span_ms)
    print("  GPU 忙碌合计          %.1f ms" % busy_ms)
    print("  >>> GPU 在时间线上的占用率  %.1f%%  （这是真实的，不是 Windows 计数器）"
          % (100.0 * busy_ms / max(span_ms, 1e-9)))
    print("  时间片中「有 kernel 在跑」的比例  %.1f%%" % (100.0 * occ.mean()))
    print("  GPU 事件个数          %d" % len(ev))
    print()
    print("  判读：占用率低 + 片占用高  ⇒ GPU 在跑但每个 kernel 很短（碎片化）")
    print("        占用率低 + 片占用低  ⇒ 有明显空洞（CPU 喂不上 / 依赖串行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
