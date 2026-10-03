---
uid: 9e3a0022
id: neko-vrc-body.config.io-sections
parent: neko-vrc-body.config
name: {zh: "设备接入配置段", en: "Device IO Sections"}
description:
  zh: >
      OSC、聊天框中继、驱动遥测与虚拟控制器的外部端点与安全限制。
      
  en: >
      External endpoints and safety limits for OSC, the chatbox relay, driver telemetry and the virtual controllers.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.813Z"
fingerprint: 55757bfaa420ee4e8c4479614d38026395bc79552cfe74aec0d249ef070ee23f
source:
  - path: "config.py"
    line: 125
    end_line: 195
apis:
  - protocol: file
    path: "config.py#VrchatOscConfig"
    description:
      zh: >
          VRChat OSC 收发地址、聊天框与输入脉冲限制。
          
      en: >
          VRChat OSC listen/send addresses, chatbox and input-pulse limits.
          
  - protocol: file
    path: "config.py#ChatboxRelayConfig"
    description:
      zh: >
          是否以及如何把宿主对话轮转发到 VRChat 聊天框。
          
      en: >
          Whether and how host conversation turns are forwarded to the VRChat chatbox.
          
  - protocol: file
    path: "config.py#DriverLogConfig"
    description:
      zh: >
          驱动遥测组播组、协议版本与 JSONL 落盘路径。
          
      en: >
          Driver telemetry multicast group, protocol version and JSONL sink path.
          
  - protocol: file
    path: "config.py#ControllerInputConfig"
    description:
      zh: >
          虚拟 AnyaDance 控制器输入的路由与安全限制。
          
      en: >
          Routing and safety limits for the virtual AnyaDance controller inputs.
          
---
