---
uid: 9e3f0003
id: neko-vrc-body.backend.cognition.preconditions.gate
parent: neko-vrc-body.backend.cognition.preconditions
name: {zh: "前置条件闸门", en: "Precondition Gate"}
description:
  zh: >
      依据世界状态快照检查动作前置条件，不接触实时控制线程。
      
  en: >
      Checks action preconditions against a world-state snapshot without touching the live control thread.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.588Z"
fingerprint: ef89d358bc9669d29d55dc2f3e13d67ab8ffb2d1c9f55ccdf7de4175419d1d8a
source:
  - path: "backend/cognition.py"
    line: 220
    end_line: 438
apis:
  - protocol: file
    path: "backend/cognition.py#WorldPreconditionGate.evaluate"
    description:
      zh: >
          依据世界状态快照判定动作前置条件是否成立。
          
      en: >
          Decides against a world-state snapshot whether an action's preconditions hold.
          
  - protocol: file
    path: "backend/cognition.py#WorldPreconditionGate.evaluate_normalized"
    description:
      zh: >
          在已归一化的约束上执行判定。
          
      en: >
          Runs the evaluation on the already normalized constraints.
          
  - protocol: file
    path: "backend/cognition.py#WorldPreconditionGate.normalize"
    description:
      zh: >
          把输入归一化为可判定的标准约束形式。
          
      en: >
          Normalizes input into the canonical constraint form used for evaluation.
          
---
