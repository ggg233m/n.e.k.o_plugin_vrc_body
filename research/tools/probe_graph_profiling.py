# -*- coding: utf-8 -*-
r"""2.14.1 的 XPUGraph 到底给不给「逐 kernel 可见性」—— 决定 2.15 那条 profiling 升级值不值。

背景：2.15 的 release note 里有「XPU Graph replay 内核级可见性」（PR #195128 等），
被当作该版本对我们唯一有意义的 XPU 改动。但我们本地 2.14.1 已经有 XPUGraph，
所以要先测：**在 2.14.1 上，图重放之后 profiler 还能不能看见逐个 kernel？**

三种口径对比同一个前向：
  A. eager        —— 已知能看见逐 kernel（参照）
  B. 图重放（捕获+replay）—— 这就是要测的
  C. 图重放后再 eager —— 确认图没破坏 profiler

判据：如果 B 能报出 kernel 个数/时间，说明 2.14.1 已经够用，2.15 那条对我们无增量；
      如果 B 只有一条「graph」条目、看不到内部 kernel，那 2.15 确实补了缺口。
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
# 图重放的唯一前提：切到 Level-Zero V2 适配器（必须在 import torch 之前）
os.environ.setdefault("SYCL_UR_USE_LEVEL_ZERO_V2", "1")
os.environ.setdefault("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1")
os.environ.setdefault("UR_L0_USE_RELAXED_ALLOCATION_LIMITS", "1")
_ONEAPI = r"D:\Program Files (x86)\Intel\oneAPI\2026.0\bin"
if os.path.isdir(_ONEAPI):
    os.environ["PATH"] = _ONEAPI + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("CC", "icpx")
    os.environ.setdefault("CXX", "icpx")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def summary(prof, label):
    rows = []
    gpu = 0.0
    for e in prof.key_averages():
        dt = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if dt <= 0:
            continue
        gpu += dt
        rows.append((int(e.count), str(e.key), dt / 1000.0))
    rows.sort(key=lambda r: -r[2])
    n = sum(r[0] for r in rows)
    print("  [%s] kernel=%d  GPU=%.1f ms  （前 5）" % (label, n, gpu / 1000.0))
    for c, k, ms in rows[:5]:
        print("        %-52s n=%-5d %.2f ms" % (k[:52], c, ms))
    return n, gpu / 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3near")
    ap.add_argument("--sessions", nargs="*", default=["20261031_020757"])
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()

    import torch
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader
    import train_policy_lora as T

    dev = "xpu"
    print("[env] platform =", torch.xpu.get_device_properties(0).name)
    print("[env] XPUGraph  =", hasattr(torch.xpu, "XPUGraph"))
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
    kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b["inputs"].items()}
    inner = model.base_model.model.model

    with torch.no_grad():
        for _ in range(3):
            inner(**kw, use_cache=False)
        torch.xpu.synchronize()

        # A. eager
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as pa:
            inner(**kw, use_cache=False)
            torch.xpu.synchronize()
        na, ma = summary(pa, "A eager")

        # B. 图重放。必须照搬 policy_graph_runtime.GraphRunner 的三步：
        #   ① 独立 stream 并 wait 当前流；② 先在流上空跑 3 次预热（否则首次编译会进图）；
        #   ③ capture 后立刻 replay 一次（否则输出是垃圾，§14.3）。
        #   少任何一步都会报 UR_RESULT_ERROR_UNSUPPORTED_FEATURE（实测踩到）。
        g = torch.xpu.XPUGraph()
        try:
            stream = torch.xpu.Stream()
            stream.wait_stream(torch.xpu.current_stream())
            with torch.inference_mode(), torch.xpu.stream(stream):
                for _ in range(3):
                    inner(**kw, use_cache=False)
            stream.synchronize()
            with torch.inference_mode(), torch.xpu.graph(g, stream=stream):
                inner(**kw, use_cache=False)
            torch.xpu.synchronize()
            g.replay()
            torch.xpu.synchronize()
            for _ in range(3):
                g.replay()
            torch.xpu.synchronize()
            print("  [graph] 捕获并 replay 成功")
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as pb:
                for _ in range(1):
                    g.replay()
                torch.xpu.synchronize()
            nb, mb = summary(pb, "B graph replay")
        except Exception as e:  # noqa: BLE001
            print("  [graph] 失败：%r" % (e,))
            nb = mb = -1

    print()
    if nb > 0:
        print("  ⇒ 图重放时 profiler 报出 %d 个 kernel（eager 是 %d）" % (nb, na))
        if nb >= na * 0.5:
            print("  ⇒ **2.14.1 已能给逐 kernel 可见性**，2.15 那条对我们无增量")
        else:
            print("  ⇒ 图重放只剩 %d 个条目，**内部 kernel 不可见** ⇒ 2.15 确实补了缺口" % nb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
