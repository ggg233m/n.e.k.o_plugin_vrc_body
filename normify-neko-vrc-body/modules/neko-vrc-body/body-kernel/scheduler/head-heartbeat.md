---
uid: 9e3a0329
id: neko-vrc-body.body-kernel.scheduler.head-heartbeat
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "头部心跳", en: "Head Heartbeat"}
description:
  zh: >
      周期性的只带头显心跳：身体空闲时靠它维持虚拟 HMD 偏航。
      
  en: >
      The periodic head-only heartbeat that keeps the virtual HMD yaw alive when the body is otherwise idle.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.793Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 852
    end_line: 855
  - path: "scheduler.py"
    line: 1490
    end_line: 1503
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._head_heartbeat_due"
    description:
      zh: >
          周期性的只带头显心跳是否到期。
          
      en: >
          Whether the periodic head-only heartbeat is due.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._send_head_frame"
    description:
      zh: >
          发送只带头显的那一帧，维持虚拟 HMD 偏航。
          
      en: >
          Send the head-only frame that keeps virtual HMD yaw alive.
          
---
