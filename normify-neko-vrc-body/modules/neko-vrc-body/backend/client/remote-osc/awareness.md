---
uid: 9e3c010d
id: neko-vrc-body.backend.client.remote-osc.awareness
parent: neko-vrc-body.backend.client.remote-osc
name: {zh: "状态与参数", en: "Awareness and Parameters"}
description:
  zh: >
      OSC 代理的状态、aware 读取与化身参数写入。
      
  en: >
      OSC proxy state, awareness reads and avatar parameter writes.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.582Z"
fingerprint: 8f6d5a4ca8fb37bed50fe66446dfcb2eb54d67a9caca0125f881ac24d9844a7e
source:
  - path: "backend/client.py"
    line: 464
    end_line: 498
apis:
  - protocol: file
    path: "backend/client.py#RemoteOsc.awareness"
    description:
      zh: >
          读取远端 OSC 代理的 awareness 状态。
          
      en: >
          Reads the awareness state of the remote OSC agent.
          
  - protocol: file
    path: "backend/client.py#RemoteOsc.send_parameter"
    description:
      zh: >
          向远端 OSC 代理写入一个化身参数。
          
      en: >
          Writes one avatar parameter on the remote OSC agent.
          
---
