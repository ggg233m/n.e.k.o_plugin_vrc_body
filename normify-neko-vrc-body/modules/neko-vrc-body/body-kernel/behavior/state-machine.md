---
uid: 9e3a0308
id: neko-vrc-body.body-kernel.behavior.state-machine
parent: neko-vrc-body.body-kernel.behavior
name: {zh: "行为状态机", en: "Behavior State Machine"}
description:
  zh: >
      调度器持有的分层状态机：基础动作、叠加层、拒绝与有界迁移历史。
      
  en: >
      The layered state machine the scheduler owns: base actions, overlays, rejections and a bounded transition history.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.782Z"
fingerprint: 1a12074fd870f4fbc2e33a1a6015fbc4a415817d225662e7f7d1613b78c7a42a
source:
  - path: "behavior.py"
    line: 120
    end_line: 286
apis:
  - protocol: file
    path: "behavior.py#BehaviorStateMachine"
    description:
      zh: >
          调度器持有的基础/叠加行为状态，带有界的迁移历史。
          
      en: >
          Scheduler-owned base/overlay behaviour state with bounded transition history.
          
  - protocol: file
    path: "behavior.py#BehaviorStateMachine.activate_base"
    description:
      zh: >
          启动一个基础层动作。
          
      en: >
          Start a base-layer action.
          
  - protocol: file
    path: "behavior.py#BehaviorStateMachine.snapshot"
    description:
      zh: >
          为 aware 负载渲染当前行为状态。
          
      en: >
          Render the current behaviour state for the awareness payload.
          
---
