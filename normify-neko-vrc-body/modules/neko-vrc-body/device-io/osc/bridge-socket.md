---
uid: 9e3a0342
id: neko-vrc-body.device-io.osc.bridge-socket
parent: neko-vrc-body.device-io.osc
name: {zh: "Socket 归属", en: "Socket Ownership"}
description:
  zh: >
      socket 归属、线程生命周期，以及带错误记录的低层发送路径。
      
  en: >
      Socket ownership, thread lifecycle and the low-level send path with error recording.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.828Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 243
    end_line: 364
apis:
  - protocol: file
    path: "osc.py#VrchatOscBridge._send"
    description:
      zh: >
          向配置的 VRChat 端点发送一条原始 OSC 报文。
          
      en: >
          Send a raw OSC message to the configured VRChat endpoint.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge.start"
    description:
      zh: >
          启动收发线程。
          
      en: >
          Start the send and receive threads.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge.stop"
    description:
      zh: >
          停止两个线程并释放 socket。
          
      en: >
          Stop both threads and release the sockets.
          
---
