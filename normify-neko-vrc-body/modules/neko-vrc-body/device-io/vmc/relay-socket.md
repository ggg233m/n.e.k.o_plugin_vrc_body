---
uid: 9e3a0353
id: neko-vrc-body.device-io.vmc.relay-socket
parent: neko-vrc-body.device-io.vmc
name: {zh: "中继 Socket 与接收", en: "Relay Socket and Ingest"}
description:
  zh: >
      VMC 中继的线程生命周期与数据报/消息接收入口。
      
  en: >
      Thread lifecycle and the packet/message ingestion entry points of the VMC relay.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.834Z"
fingerprint: a49ba23fb5ec0e9ebaa1ef04e8b0a3ded0b99c3a792d7b1568bce88cb271c578
source:
  - path: "vmc_idle.py"
    line: 295
    end_line: 426
apis:
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay.start"
    description:
      zh: >
          启动 VMC 接收线程。
          
      en: >
          Start the VMC receive thread.
          
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay.stop"
    description:
      zh: >
          停止 VMC 接收线程。
          
      en: >
          Stop the VMC receive thread.
          
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay.ingest_packet"
    description:
      zh: >
          接收一个原始 VMC 数据报。
          
      en: >
          Ingest one raw VMC datagram.
          
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay.ingest_messages"
    description:
      zh: >
          接收已解码的 VMC 消息。
          
      en: >
          Ingest already-decoded VMC messages.
          
---
