---
uid: 9e3b000a
id: neko-vrc-body.tests.device.chatbox-relay
parent: neko-vrc-body.tests.device
name: {zh: "会话转聊天框中继轮询与去重测试", en: "Conversation-to-Chatbox Relay Polling and Dedup Tests"}
description:
  zh: >
      覆盖会话消息转聊天框的中继轮询、去重逻辑与发送节流。
      
  en: >
      Covers relay polling from conversation to chatbox, de-duplication logic, and send throttling.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.920Z"
fingerprint: c8eda9a1dfc3e82a70ba9b5e2ddfd1ac6be22e1fe02a4f1e53b50a2715dab0e4
source:
  - path: "tests/test_chatbox_relay.py"
apis:
  - protocol: file
    path: "tests/test_chatbox_relay.py"
    description:
      zh: >
          会话转聊天框中继轮询与去重的测试文件。
          
      en: >
          Test file for conversation-to-chatbox relay polling and de-duplication.
          
---
