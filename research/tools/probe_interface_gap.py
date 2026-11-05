# -*- coding: utf-8 -*-
"""训练/推理接口 vs GameScaling 官方口径的差距体检。

只做**只读检查**，不训练、不改模型。输出四项官方要素的现状：
  1. 动作块长度（官方 200 ms = 6 x 33 ms）
  2. history turn（官方 5 帧历史，图+动作）
  3. 结构化动作格式（官方 "X Y Z ; k1 k2 k3 ; ..."）
  4. 输出通道（官方走 lm_head 词表 + <|action_start|> 标签）
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, "research/tools")

MODEL = Path("G:/vrc_recording/vrc_model/weights/GameScaling-0.8B-S3")

print("=" * 74)
print("  GameScaling 官方口径 vs 当前实现")
print("=" * 74)


def row(name, official, ours, verdict, note=""):
    print("  %-16s %-22s %-24s %s" % (name, official, ours, verdict))
    if note:
        print("  %-16s %s" % ("", note))


# ---- 1. 动作块 ----
import train_policy_lora as T
# 训练脚本只有 K_STEPS，步长写在数据集构建侧（build_policy_dataset.STEP_S）
import build_policy_dataset as B
row("动作块", "6 x 33ms = 200ms",
    "%d x %dms = %dms" % (T.K_STEPS, int(B.STEP_S * 1000),
                           int(T.K_STEPS * B.STEP_S * 1000)),
    "不匹配")

# ---- 2. history turn ----
rt_src = Path("research/tools/policy_runtime.py").read_text(encoding="utf-8")
has_hist = "history" in rt_src.lower() or "past_frame" in rt_src.lower()
row("历史轮", "5 帧（图+动作）", "0 帧（只有 prev+cur 两帧视频）",
    "不匹配" if not has_hist else "部分")

# ---- 3. 结构化动作 ----
pt_src = Path("research/tools/policy_text.py").read_text(encoding="utf-8")
row("动作格式", "结构化文本键鼠", "连续 (vx,vz,omega) 向量", "不匹配")

# ---- 4. 输出通道 ----
tr_src = Path("research/tools/train_policy_lora.py").read_text(encoding="utf-8")
uses_lmhead = "lm_head" in tr_src
uses_tag = "action_start" in tr_src
print("  %-16s %-22s %-24s %s" % ("输出通道", "lm_head + <|action_start|>",
      ("lm_head" if uses_lmhead else "外挂 Linear(1024->36)"),
      "匹配" if uses_lmhead else "不匹配"))
print("  %-16s %s" % ("", "训练脚本里出现 action_start 标签: %s" % uses_tag))


# ---- 官方 chat template 是否支持 action 标签 ----
tj = (MODEL / "chat_template.jinja")
if tj.exists():
    s = tj.read_text(encoding="utf-8")
    print("  %-16s chat_template 含 action_start: %s" % ("模板", "action_start" in s))


# ---- 词表里关键 token 是否单 token ----
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
probe = ["<|action_start|>", "<|action_end|>", "<|video_pad|>",
         "W", "A", "S", "D", ";", "three"]
print()
print("  token 化检查（是否为单 token）:")
for p in probe:
    ids = tok.encode(p, add_special_tokens=False)
    print("    %-20s %-16s %s" % (p, ids, "单token OK" if len(ids) == 1 else "%d tokens" % len(ids)))
