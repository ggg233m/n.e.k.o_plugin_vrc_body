---
uid: 9e3a023a
id: neko-vrc-body.plugin.tools.chatbox-relay
parent: neko-vrc-body.plugin.tools
name: {zh: "聊天框中继开关", en: "Chatbox Relay Switch"}
description:
  zh: >
      开关「把角色说出口的那句自动转发到 VRChat 聊天框」的中继。
      
  en: >
      Toggling the automatic relay that forwards the character's spoken lines into the VRChat chatbox.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.895Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2820
    end_line: 2844
apis:
  - protocol: rpc
    path: "body_chatbox_relay_enable"
    description:
      zh: >
          工具：启用把角色发言自动转发到聊天框。
          
      en: >
          Tool: enable automatic forwarding of the character's speech to the chatbox.
          
  - protocol: rpc
    path: "body_chatbox_relay_disable"
    description:
      zh: >
          工具：关闭聊天框自动转发。
          
      en: >
          Tool: disable automatic chatbox forwarding.
          
---
