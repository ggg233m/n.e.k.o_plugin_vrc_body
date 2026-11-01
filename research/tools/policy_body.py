# -*- coding: utf-8 -*-
"""policy 的 body 反馈策略：OSC 回传优先，静止卡死时冷启动引导。

2026-11-01 在线首跑实测（静止帧 x body 四档）：body=0/0.5/0.8/1.0 m/s 时模型输出
0.08/0.44/1.97/0.72 —— 模型把 body 当主导特征。训练数据里人一直在走、body 与
画面光流一致；闭环冷启动时角色静止 → body=0 → 输出≈0 → 角色不动 → body 仍=0，
锁死。

**引导触发条件（两版教训）**：不能只看"OSC 静默"—— VRChat 对静止角色每帧都回传
velocity=0，available=True、值=0，引导永远进不去。正确判据是"回传速度≈0 且航点
还在远处"：想走而没在走，才推一把。一旦回传速度上来（真在走了），立刻切回真实值。
"""
from __future__ import annotations

NUDGE_SPEED = 0.6      # 引导速度上限 m/s（v_max=1.2 的一半）
NUDGE_MIN = 0.2        # 航点再近也至少给这个速度，避免 0 除
STALL_SPEED = 0.15     # 回传速度低于此值视为"没在走"（m/s）
STALL_WP_M = 0.3       # 航点距离超过此值才引导（再近就该到了）


def body_feedback(osc_body: tuple[float, float] | None, wp, nudged: bool = True) -> tuple[float, float]:
    """返回该拍喂给模型的 body。

    osc_body：OSC 最近回传（ZOH 窗口内，可能全 0）或 None。
    wp：机体系航点 (dx, dz) 或 None。
    """
    dx = dz = 0.0
    has_wp = False
    if wp:
        dx, dz = float(wp[0]), float(wp[1])
        has_wp = (dx * dx + dz * dz) ** 0.5 > 1e-3

    if osc_body is not None:
        speed = (osc_body[0] ** 2 + osc_body[1] ** 2) ** 0.5
        # 真的在走 → 用真实值。没在走且航点还远 → 引导（破静止死锁）。
        if speed >= STALL_SPEED or not has_wp or not nudged:
            return osc_body

    if not (nudged and has_wp):
        return (0.0, 0.0) if osc_body is None else osc_body
    dist = (dx * dx + dz * dz) ** 0.5
    if dist < STALL_WP_M:
        return osc_body if osc_body is not None else (0.0, 0.0)
    v = min(NUDGE_SPEED, max(NUDGE_MIN, dist))
    return (v * dx / dist, v * dz / dist)