---
uid: 9e3a0306
id: neko-vrc-body.body-kernel.behavior.policies
parent: neko-vrc-body.body-kernel.behavior
name: {zh: "运动策略", en: "Motion Policies"}
description:
  zh: >
      按种类的运动策略、默认策略表、表情意图与纯头部手势集合。
      
  en: >
      Per-kind motion policies, the default policy table, expression intents and the head-only gesture set.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.782Z"
fingerprint: 1a12074fd870f4fbc2e33a1a6015fbc4a415817d225662e7f7d1613b78c7a42a
source:
  - path: "behavior.py"
    line: 12
    end_line: 78
apis:
  - protocol: file
    path: "behavior.py#MotionPolicy"
    description:
      zh: >
          某个身体命令种类的运动策略：时长上下界、混合与叠加优先级。
          
      en: >
          Motion policy for a body command kind: duration bounds, blending and overlay priority.
          
  - protocol: file
    path: "behavior.py#policy_for"
    description:
      zh: >
          解析某个命令种类的运动策略。
          
      en: >
          Resolve the motion policy for a command kind.
          
  - protocol: file
    path: "behavior.py#HEAD_ONLY_GESTURES"
    description:
      zh: >
          绝不干扰手臂的纯头部手势集合。
          
      en: >
          Head-only gestures that never disturb the arms.
          
---
