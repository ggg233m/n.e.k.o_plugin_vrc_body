---
uid: 9e3a0320
id: neko-vrc-body.body-kernel.scheduler.transport
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "传输层契约", en: "Transport Contracts"}
description:
  zh: >
      调度器依赖的传输层与空闲帧源协议，以及默认 UDP 实现。
      
  en: >
      The transport and idle-frame-source protocols the scheduler depends on, and the default UDP implementation.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.802Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 50
    end_line: 79
apis:
  - protocol: file
    path: "scheduler.py#DatagramTransport"
    description:
      zh: >
          调度器所需的传输层契约：发送、关闭与本地端口。
          
      en: >
          The transport contract the scheduler needs: send, close and local port.
          
  - protocol: file
    path: "scheduler.py#IdleFrameSource"
    description:
      zh: >
          空闲帧源契约：最新帧加一份快照。
          
      en: >
          Idle frame source contract: latest frame plus a snapshot.
          
  - protocol: file
    path: "scheduler.py#UdpTransport"
    description:
      zh: >
          默认的 UDP 数据报传输层。
          
      en: >
          The default UDP datagram transport.
          
---
