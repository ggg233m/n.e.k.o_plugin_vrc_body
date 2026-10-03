---
uid: 9e3a0325
id: neko-vrc-body.body-kernel.scheduler.input-axes
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "控制器输入轴", en: "Controller Input Axes"}
description:
  zh: >
      轴、按键与叠加层处理：工具驱动的虚拟控制器层，以及它如何合入出站帧。
      
  en: >
      Axis, button and overlay handling: the virtual controller layer the tools drive, and how it merges into outgoing frames.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.795Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 758
    end_line: 788
  - path: "scheduler.py"
    line: 790
    end_line: 822
  - path: "scheduler.py"
    line: 912
    end_line: 927
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._apply_input_axes_command"
    description:
      zh: >
          把输入轴命令施加到控制器叠加层。
          
      en: >
          Apply an input-axes command to the controller overlay.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._apply_input_button_command"
    description:
      zh: >
          把输入按键命令施加到控制器叠加层。
          
      en: >
          Apply an input-button command to the controller overlay.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._apply_controller_input_overlay"
    description:
      zh: >
          把控制器叠加层合入即将发送的帧。
          
      en: >
          Merge the controller overlay into the frame about to be sent.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler.controller_input_snapshot"
    description:
      zh: >
          为状态回读提供控制器叠加层快照。
          
      en: >
          Read the controller overlay for the status read-out.
          
---
