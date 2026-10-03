---
uid: 9e3a0371
id: neko-vrc-body.device-io.chatbox-relay.lifecycle
parent: neko-vrc-body.device-io.chatbox-relay
name: {zh: "中继生命周期", en: "Relay Lifecycle"}
description:
  zh: >
      后台轮询器的构造、启用开关与启动/停止生命周期。
      
  en: >
      Construction, the enable switch and the start/stop lifecycle of the background poller.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.817Z"
fingerprint: 8446eceac711a656f2cb14da577f38da57d62b7c6a8ec1866e9f0f6a1f22df34
source:
  - path: "chatbox_relay.py"
    line: 97
    end_line: 194
  - path: "chatbox_relay.py"
    line: 328
    end_line: 345
apis:
  - protocol: file
    path: "chatbox_relay.py#ChatboxRelay.start"
    description:
      zh: >
          启动轮询线程。
          
      en: >
          Start the polling thread.
          
  - protocol: file
    path: "chatbox_relay.py#ChatboxRelay.stop"
    description:
      zh: >
          停止轮询线程。
          
      en: >
          Stop the polling thread.
          
  - protocol: file
    path: "chatbox_relay.py#ChatboxRelay.set_enabled"
    description:
      zh: >
          运行时开启或关闭中继。
          
      en: >
          Turn the relay on or off at runtime.
          
  - protocol: file
    path: "chatbox_relay.py#ChatboxRelay.snapshot"
    description:
      zh: >
          报告中继的当前状态。
          
      en: >
          Report the relay's current state.
          
---
