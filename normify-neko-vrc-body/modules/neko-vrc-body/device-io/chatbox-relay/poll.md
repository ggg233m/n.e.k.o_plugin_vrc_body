---
uid: 9e3a0372
id: neko-vrc-body.device-io.chatbox-relay.poll
parent: neko-vrc-body.device-io.chatbox-relay
name: {zh: "轮询过程", en: "Polling Pass"}
description:
  zh: >
      一次轮询：只提取角色真正说出口的内容、去重，并适配聊天框长度预算。
      
  en: >
      One polling pass: extract only what the character actually said, de-duplicate, and fit the chatbox length budget.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.818Z"
fingerprint: 8446eceac711a656f2cb14da577f38da57d62b7c6a8ec1866e9f0f6a1f22df34
source:
  - path: "chatbox_relay.py"
    line: 198
    end_line: 324
apis:
  - protocol: file
    path: "chatbox_relay.py#ChatboxRelay._poll_once"
    description:
      zh: >
          对宿主对话轮执行一次轮询。
          
      en: >
          One polling pass over the host conversation turns.
          
  - protocol: file
    path: "chatbox_relay.py#ChatboxRelay._extract"
    description:
      zh: >
          从宿主记录中提取可发送的一行，或者什么都不提取。
          
      en: >
          Extract a sendable line from a host record, or nothing.
          
  - protocol: file
    path: "chatbox_relay.py#ChatboxRelay._format"
    description:
      zh: >
          把一行文本格式化为符合 VRChat 聊天框长度预算的样子。
          
      en: >
          Format a line to fit the VRChat chatbox length budget.
          
  - protocol: file
    path: "chatbox_relay.py#ChatboxRelay._remember"
    description:
      zh: >
          记住一个记录 id，已发送过的将被拒绝。
          
      en: >
          Remember a record id, rejecting one that was already sent.
          
---
