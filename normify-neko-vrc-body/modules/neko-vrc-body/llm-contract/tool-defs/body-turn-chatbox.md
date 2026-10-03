---
uid: 9e3a0104
id: neko-vrc-body.llm-contract.tool-defs.body-turn-chatbox
parent: neko-vrc-body.llm-contract.tool-defs
name: {zh: "转向与聊天框工具", en: "Turn and Chatbox Tools"}
description:
  zh: >
      相对转向与直接用聊天框说话的 Schema。
      
  en: >
      Schemas for relative turning and direct chatbox speech.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.838Z"
fingerprint: 348077cad7b4ebac68be08404f65a6094ebab38baa302e5e51c5b85edc242009
source:
  - path: "tool_defs.py"
    line: 493
    end_line: 537
apis:
  - protocol: file
    path: "tool_defs.py#BODY_TURN"
    description:
      zh: >
          工具 Schema：按相对偏航角转动化身。
          
      en: >
          Tool schema: turn the avatar by a relative yaw.
          
  - protocol: file
    path: "tool_defs.py#BODY_CHATBOX"
    description:
      zh: >
          工具 Schema：向 VRChat 聊天框发送一行文本。
          
      en: >
          Tool schema: send a line to the VRChat chatbox.
          
---
