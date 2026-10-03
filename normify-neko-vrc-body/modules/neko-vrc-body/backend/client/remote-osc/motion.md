---
uid: 9e3c010e
id: neko-vrc-body.backend.client.remote-osc.motion
parent: neko-vrc-body.backend.client.remote-osc
name: {zh: "运动与输入", en: "Motion and Input"}
description:
  zh: >
      OSC 代理的运动与输入：移动、转向、批量、聊天框与取消调度输入。
      
  en: >
      OSC proxy motion and input: locomotion, turning, batching, chatbox and cancelling scheduled inputs.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.583Z"
fingerprint: 8f6d5a4ca8fb37bed50fe66446dfcb2eb54d67a9caca0125f881ac24d9844a7e
source:
  - path: "backend/client.py"
    line: 500
    end_line: 569
apis:
  - protocol: file
    path: "backend/client.py#RemoteOsc.set_locomotion"
    description:
      zh: >
          设置远端移动速度矢量。
          
      en: >
          Sets the remote locomotion velocity vector.
          
  - protocol: file
    path: "backend/client.py#RemoteOsc.batch"
    description:
      zh: >
          向远端代理批量提交一组 OSC 命令。
          
      en: >
          Submits a batch of OSC commands to the remote agent.
          
  - protocol: file
    path: "backend/client.py#RemoteOsc.send_chatbox"
    description:
      zh: >
          通过远端 OSC 代理发送聊天框消息。
          
      en: >
          Sends a chatbox message through the remote OSC agent.
          
---
