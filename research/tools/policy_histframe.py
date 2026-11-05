# -*- coding: utf-8 -*-
"""训练/推理共用的视觉历史帧口径（唯一真源）。

为什么要有这个模块（2026-11-05）：
  历史上「当前帧 + t−0.4 s 前帧」这个间隔在 6 个文件里各写一遍字面量 0.4，
  改一次要改六处，漏一处就会训练/推理口径不一致（静默错误，形状全对）。

为什么从 0.4 s 改成 0.2 s：
  实测（`.tmp/freq_visual_gap_probe.py`，10 场）：
    帧间隔 0.2 s → 转向格子端点漏检 17.8%
    帧间隔 0.4 s → 转向格子端点漏检 26.9%   ← 旧值
  0.2 s 与 5 Hz 观测网格（GRID_S=0.2）严格对齐，即「往前 1 个样本」；
  0.4 s 是往前 2 个样本，会跳过中间那一帧，正是漏检的来源。

⚠️ 只改**视觉历史帧间隔**，不改动作标签（K_STEPS×STEP_S=12×0.1 s）、
   不改观测定点网格（GRID_S）、不改执行器步长。
"""
from __future__ import annotations

# 视觉历史帧间隔（秒）：当前帧与历史帧的时间差。
HISTORY_FRAME_S = 0.2

# 观测定点网格步长（秒）。5 Hz。与 HISTORY_FRAME_S 对齐后 = 往前 1 个样本。
OBS_GRID_S = 0.2

# 在观测定点网格上，历史帧相对当前帧往回退几个样本。
HISTORY_FRAME_STEPS = int(round(HISTORY_FRAME_S / OBS_GRID_S))


def history_frame_offset_s() -> float:
    """返回历史帧的负向时间偏移（秒），供各处 `t - offset` 使用。"""
    return HISTORY_FRAME_S


def describe() -> str:
    return "history_frame=%.2fs (%d step on %.2fs grid)" % (
        HISTORY_FRAME_S, HISTORY_FRAME_STEPS, OBS_GRID_S)
