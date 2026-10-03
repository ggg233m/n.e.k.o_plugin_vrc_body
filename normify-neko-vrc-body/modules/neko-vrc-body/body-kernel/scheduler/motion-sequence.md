---
uid: 9e3a032c
id: neko-vrc-body.body-kernel.scheduler.motion-sequence
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "动作序列", en: "Motion Sequence"}
description:
  zh: >
      多段动作序列，带逐段截止时间与采样闭包。
      
  en: >
      Multi-segment motion sequences with per-segment deadlines and sampling closures.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.798Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 1181
    end_line: 1280
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._start_sequence"
    description:
      zh: >
          启动带逐段时序的多段动作序列。
          
      en: >
          Start a multi-segment motion sequence with per-segment timing.
          
---
