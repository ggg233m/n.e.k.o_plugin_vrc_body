# -*- coding: utf-8 -*-
r"""找出 dynamo 在哪里断图 —— 那 1188 个「ATen 原样调用」从哪来。

已知（2026-11-03 实测）：eager 9892 个算子 → compile 后 3715 个 kernel，融合率 62.4%。
剩余 3715 里 1188 个是「ATen 原样调用」（没被 Inductor 接管）、783 个是库 GEMM（正常）。
最大嫌疑：18 层 GatedDeltaNet 的参考实现里有 data-dependent 的 Python 循环
（transformers `torch_recurrent_gated_delta_rule` 的 `for i in range(sequence_length)`），
dynamo 到那里就断图，之后的算子全留在图外。

本脚本用 torch._dynamo.explain() 的图断点统计给出**断在什么原因上、几处**，
并输出每个子图的算子数，用来判断「还有没有办法把它们拉进图」。
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
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--capture-scalars", action="store_true",
                    help="torch._dynamo.config.capture_scalar_outputs=True")
    ap.add_argument("--static", action="store_true",
                    help="dynamic=False（assume_static_by_default）")
    a = ap.parse_args()

    import torch
    import torch._dynamo as dynamo
    if a.capture_scalars:
        dynamo.config.capture_scalar_outputs = True
        print("[cfg] capture_scalar_outputs=True", flush=True)
    if a.static:
        dynamo.config.assume_static_by_default = True
        print("[cfg] assume_static_by_default=True", flush=True)
    from transformers import AutoProcessor
    from torch.utils.data import DataLoader
    import train_policy_lora as T

    dev = "xpu"
    proc = AutoProcessor.from_pretrained("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")
    px = 448 * 256
    samples = T.build_index(Path(a.dataset), a.sessions)
    samples = [s for s in samples if s["split"] == "train"][: a.batch * 3]
    ds = T.PolicyDataset(samples, proc, px, px, training=True)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0,
                    collate_fn=T.make_collate(proc, px, px))
    b = next(iter(dl))

    act_dim = int(np.asarray(samples[0]["act"]).shape[-1])
    model, head, hidden = T.load_model("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3",
                                       128, dev, act_dim=act_dim)
    model.eval()

    inner = model.base_model.model.model

    inputs = {k: v for k, v in b["inputs"].items()}
    # explain() 只接受位置/关键字实参，不展开 dict；且 inputs 里的张量必须已上设备
    # （train 里是 forward_batch 内部搬的，这里手动搬，否则报 Tensor device mismatch）
    inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    expl = dynamo.explain(inner, **inputs)
    print()
    print("  graph_break_count      %d" % expl.graph_break_count)
    print("  op_count               %d" % expl.op_count)
    for attr in ("unique_graphs", "graph_count", "num_graphs"):
        if hasattr(expl, attr):
            print("  %-22s %d" % (attr, getattr(expl, attr)))
            break
    print("  attrs                  %r" % (sorted(a for a in dir(expl) if not a.startswith("_")),))
    print()
    reasons = getattr(expl, "break_reasons", None)
    if reasons:
        from collections import Counter
        c = Counter(str(getattr(r, "reason", r))[:150] for r in reasons)
        print("  断图原因（按次数）：")
        for reason, n in c.most_common(15):
            print("    [%2d] %s" % (n, reason))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
