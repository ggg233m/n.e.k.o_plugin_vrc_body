---
uid: 9e3a0208
id: neko-vrc-body.plugin.conversation.send-line
parent: neko-vrc-body.plugin.conversation
name: {zh: "发送聊天框文本", en: "Send Chatbox Line"}
description:
  zh: >
      通过 OSC 桥把一行格式化文本送进 VRChat 聊天框。
      
  en: >
      Forwarding a single formatted line into the VRChat chatbox through the OSC bridge.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.864Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 370
    end_line: 379
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._send_chatbox_line"
    description:
      zh: >
          向 VRChat 聊天框发送一行文本，回报成功与传输错误。
          
      en: >
          Send one line to the VRChat chatbox, reporting success and any transport error.
          
deps:
  - kind: call
    to: neko-vrc-body.device-io.osc.bridge-parameter
    label: {zh: "通过 OSC 在聊天框输入", en: "Type into the chatbox over OSC"}
---
