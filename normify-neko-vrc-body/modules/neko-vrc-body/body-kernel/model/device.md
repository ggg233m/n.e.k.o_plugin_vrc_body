---
uid: 9e3a0300
id: neko-vrc-body.body-kernel.model.device
parent: neko-vrc-body.body-kernel.model
name: {zh: "设备与控制器状态", en: "Device and Controller State"}
description:
  zh: >
      设备与控制器状态，以及调度器和协议共用的规范 id 表。
      
  en: >
      Device and controller state plus the canonical id tables shared by the scheduler and the protocol.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.784Z"
fingerprint: 172f824e6246028ebf80474b7ac6a938e3b15068a82f1e384a1b87744984bc0c
source:
  - path: "model.py"
    line: 13
    end_line: 55
apis:
  - protocol: file
    path: "model.py#DeviceState"
    description:
      zh: >
          单个被跟踪设备的位置与朝向。
          
      en: >
          One tracked device's position and orientation.
          
  - protocol: file
    path: "model.py#ControllerState"
    description:
      zh: >
          单个控制器的轴、按键与位姿。
          
      en: >
          One controller's axes, buttons and pose.
          
  - protocol: file
    path: "model.py#DEVICE_IDS"
    description:
      zh: >
          协议接受的规范设备标识。
          
      en: >
          The canonical device identifiers the protocol accepts.
          
  - protocol: file
    path: "model.py#CONTROLLER_IDS"
    description:
      zh: >
          协议接受的规范控制器标识。
          
      en: >
          The canonical controller identifiers the protocol accepts.
          
---
