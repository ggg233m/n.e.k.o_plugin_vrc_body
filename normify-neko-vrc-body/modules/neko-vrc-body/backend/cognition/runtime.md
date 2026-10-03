---
uid: 9e3f0006
id: neko-vrc-body.backend.cognition.runtime
parent: neko-vrc-body.backend.cognition
name: {zh: "认知运行时", en: "Cognition Runtime"}
description:
  zh: >
      由后端服务持有的有界状态估计、规划与反馈状态。
      
  en: >
      Bounded estimation, planning, and feedback state held by the backend service.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.589Z"
fingerprint: ef89d358bc9669d29d55dc2f3e13d67ab8ffb2d1c9f55ccdf7de4175419d1d8a
source:
  - path: "backend/cognition.py"
    line: 709
    end_line: 857
apis:
  - protocol: file
    path: "backend/cognition.py#CognitionRuntime.observe"
    description:
      zh: >
          接收一条观测并更新运行时状态估计。
          
      en: >
          Receives one observation and updates the runtime state estimate.
          
  - protocol: file
    path: "backend/cognition.py#CognitionRuntime.plan"
    description:
      zh: >
          基于当前估计状态产出归一化计划。
          
      en: >
          Produces a normalized plan from the current estimated state.
          
  - protocol: file
    path: "backend/cognition.py#CognitionRuntime.check_preconditions"
    description:
      zh: >
          对候选动作执行前置条件判定。
          
      en: >
          Evaluates the preconditions of a candidate action.
          
  - protocol: file
    path: "backend/cognition.py#CognitionRuntime.feedback"
    description:
      zh: >
          回写执行结果作为后续规划的反馈。
          
      en: >
          Writes an execution result back as feedback for later planning.
          
---
