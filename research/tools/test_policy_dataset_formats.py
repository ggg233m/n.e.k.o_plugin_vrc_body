# -*- coding: utf-8 -*-
"""回归测试：PolicyDataset 在 legacy / card 两种 prompt_format 下都能取到样本。

legacy 分支此前 _wp_in 未赋值 -> NameError（静态检查能过，运行时必炸）。
本测试用真实数据 + 真实 processor 走一遍 __getitem__。
"""
import sys
from pathlib import Path
sys.path.insert(0, "research/tools")
import train_policy_lora as T

ds = Path("G:/vrc_recording/vrc_model/dataset/m0_policy_448_v3far")
idx = [x for x in T.build_index(ds, None) if x["split"] == "val"][:2]
print("[index] val 样本 %d 条" % len(idx))

from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained(T.DEFAULT_MODEL if hasattr(T, "DEFAULT_MODEL") else "Qwen/Qwen3.5-0.8B",
                                     trust_remote_code=True)
for fmt in ("card", "legacy"):
    d = T.PolicyDataset(idx, proc, 448*256*0.25, 448*256, prompt_format=fmt, drop_wp=False)
    it = d[0]
    print("[%s] video=%s text_len=%d wp=%s num_dim=%d"
          % (fmt, tuple(it["video"].shape), len(it["text"]), it["wp"].tolist(), it["num"].shape[0]))
    assert it["text"], "text 为空"
print("OK: both formats work")
