---
uid: 9e3a0327
id: neko-vrc-body.body-kernel.scheduler.expression-overlay
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "表情叠加层", en: "Expression Overlay"}
description:
  zh: >
      在基础姿态之上启动、采样、取消与丢弃低优先级表情叠加层。
      
  en: >
      Starting, sampling, cancelling and dropping low-priority expression overlays layered onto the base pose.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.791Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 949
    end_line: 1020
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._start_expression_overlay"
    description:
      zh: >
          启动一个语义表情叠加层。
          
      en: >
          Start a semantic expression overlay.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._sample_expression_motion"
    description:
      zh: >
          每一拍采样当前的活跃表情叠加层。
          
      en: >
          Sample the active expression overlay each tick.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._drop_expression_overlays"
    description:
      zh: >
          丢弃全部表情叠加层并记录结果。
          
      en: >
          Drop every expression overlay with a recorded outcome.
          
deps:
  - kind: call
    to: neko-vrc-body.clips.expression-motion.apply
    label: {zh: "施加叠加增量", en: "Apply overlay delta"}
  - kind: call
    to: neko-vrc-body.body-kernel.behavior.expression-resolve
    label: {zh: "叠加层准入", en: "Overlay admission"}
---
