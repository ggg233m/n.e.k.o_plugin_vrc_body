---
uid: 9e3a0348
id: neko-vrc-body.device-io.osc.bridge-receive
parent: neko-vrc-body.device-io.osc
name: {zh: "接收路径", en: "Receive Path"}
description:
  zh: >
      接收侧：数据报读取、地址分发表，以及全系统读取的状态缓存。
      
  en: >
      The receive side: datagram reads, the address dispatch table and the state cache the rest of the system reads.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.826Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 767
    end_line: 858
apis:
  - protocol: file
    path: "osc.py#VrchatOscBridge._receive_once"
    description:
      zh: >
          接收一个数据报并应用到缓存的化身状态。
          
      en: >
          Receive one datagram and apply it to the cached avatar state.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge._handle_message"
    description:
      zh: >
          把一条已解码的地址/参数对应用到状态缓存。
          
      en: >
          Apply one decoded address/argument pair to the state cache.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge._run"
    description:
      zh: >
          接收线程。
          
      en: >
          The receive thread.
          
---
