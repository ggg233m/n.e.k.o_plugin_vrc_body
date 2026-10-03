---
uid: 9e3a0328
id: neko-vrc-body.body-kernel.scheduler.idle-relay
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "空闲中继与停止", en: "Idle Relay and Stop"}
description:
  zh: >
      把 VMC 中继采样为空闲基础姿态，以及安全清空动作与输入的停止路径。
      
  en: >
      Sampling the VMC relay as an idle base pose, and the stop path that clears motion and inputs safely.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.793Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 1022
    end_line: 1041
  - path: "scheduler.py"
    line: 929
    end_line: 947
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._sample_idle_relay"
    description:
      zh: >
          采样 VMC 中继的最新帧作为空闲基础姿态。
          
      en: >
          Sample the VMC relay's latest frame as the idle base pose.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._handle_stop"
    description:
      zh: >
          处理停止命令，清空动作与输入。
          
      en: >
          Handle a stop command by clearing motion and inputs.
          
deps:
  - kind: call
    to: neko-vrc-body.device-io.vmc.relay-frame
    label: {zh: "取 VMC 空闲姿态", en: "Idle pose from VMC"}
---
