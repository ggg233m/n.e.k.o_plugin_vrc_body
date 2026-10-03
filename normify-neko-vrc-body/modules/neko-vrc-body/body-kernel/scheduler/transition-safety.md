---
uid: 9e3a032f
id: neko-vrc-body.body-kernel.scheduler.transition-safety
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "过渡安全", en: "Transition Safety"}
description:
  zh: >
      最短安全过渡时长，以及每一拍末尾的帧编码与 UDP 发送。
      
  en: >
      The minimum safe transition duration, and the frame encoding plus UDP send at the end of every tick.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.801Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 1465
    end_line: 1474
  - path: "scheduler.py"
    line: 1476
    end_line: 1488
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._minimum_safe_duration"
    description:
      zh: >
          在两个姿态之间移动所需的最短安全时长。
          
      en: >
          Shortest safe duration for moving between two poses.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._send_current_frame"
    description:
      zh: >
          校验并编码当前帧，然后送上线。
          
      en: >
          Validate and encode the current frame, then put it on the wire.
          
---
