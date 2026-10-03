---
uid: 9e3a0326
id: neko-vrc-body.body-kernel.scheduler.yaw
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "偏航与转向", en: "Yaw and Turning"}
description:
  zh: >
      偏航处理：转向命令、整定判定、绝对时钟推进，以及真正转动化身的 play-space 旋转。
      
  en: >
      Yaw handling: the turn command, the settle detection, the absolute clock advance and the play-space rotation that actually turns the avatar.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.804Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 831
    end_line: 850
  - path: "scheduler.py"
    line: 857
    end_line: 910
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._apply_turn_command"
    description:
      zh: >
          启动转向命令并中止当前偏航。
          
      en: >
          Start a turn command and halt the current yaw.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._advance_yaw"
    description:
      zh: >
          把虚拟 HMD 偏航朝目标推进。
          
      en: >
          Advance the virtual HMD yaw towards its target.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._apply_play_space_yaw"
    description:
      zh: >
          把残余偏航作为 play-space 旋转施加出去。
          
      en: >
          Apply the residual yaw as a play-space rotation.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._heading_snapshot"
    description:
      zh: >
          为门控判定读取当前朝向状态。
          
      en: >
          Read the current heading state for gating decisions.
          
---
