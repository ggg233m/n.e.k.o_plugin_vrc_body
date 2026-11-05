# -*- coding: utf-8 -*-
"""把动作块按「走不走」x「转不转」正交切成 4 类。

**为什么不用原来的 idle/move/trans**（2026-11-05）：
原来的 cls 在 build_policy_dataset.py:370-375 只看平移：

    elif all(np.hypot(*act[k][:2]) <= V_STILL for k in valid):
        cls = "idle"

act[k][:2] 只取 (vx, vz)，ω 完全不参与分类。实测 idle 类里
**26.9% 的步 ω>0.3 rad/s（17 deg/s），60.9% 的样本块内有转身**，p90 的 ω=1.43 rad/s
（82 deg/s），是明确的掉头，不是噪声。

后果：copy 基线（把首步抄 8 步）在 idle 上只抄平移、ω 抄成 0，得分恒为 0；
模型只要输出任何非零 ω 就被罚。**这个格子对视觉是结构性负向偏置**——
模型越努力预测转身，这一类分数越差。用聚合数论证「视觉无用」因此不成立。

正确的切法是两个正交轴：
               w~0              w!=0
    v~0     still           turn_in_place   <- 后者视觉关键
    v!=0    walk_straight   walk_turn

判定用**块内极值**而不是均值：只要一块里有一步在走/在转，这一步的预测
就要对，均值会把「走两步停六步」平均成静止。
"""
from __future__ import annotations

import numpy as np

V_STILL = 0.1      # m/s，与 build_policy_dataset.V_STILL 一致
OMEGA_STILL = 0.3  # rad/s ~ 17 deg/s，低于此视为没转

CLASSES = ("still", "turn_in_place", "walk_straight", "walk_turn")


def classify_block(act, mask, v_still=V_STILL, omega_still=OMEGA_STILL):
    """act: (T,>=2) 前两维为平移；第 3 维存在则当作 ω(rad/s)。"""
    act = np.asarray(act, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    ok = mask > 0
    if not ok.any():
        return "still"
    a = act[ok]
    speed = np.linalg.norm(a[:, :2], axis=-1)
    moving = bool((speed > v_still).any())
    turning = bool((np.abs(a[:, 2]) > omega_still).any()) if a.shape[1] >= 3 else False
    if moving and turning:
        return "walk_turn"
    if moving:
        return "walk_straight"
    if turning:
        return "turn_in_place"
    return "still"


def relabel(samples):
    """就地给样本加 cls2 字段（不覆盖旧的 cls，方便对照）。"""
    for s in samples:
        s["cls2"] = classify_block(s["act"], s["mask"])
    return samples


def summarize(samples, key="cls2"):
    from collections import Counter
    c = Counter(s[key] for s in samples)
    n = max(sum(c.values()), 1)
    return {k: dict(n=c[k], pct=round(100.0 * c[k] / n, 1)) for k in CLASSES if c[k]}
