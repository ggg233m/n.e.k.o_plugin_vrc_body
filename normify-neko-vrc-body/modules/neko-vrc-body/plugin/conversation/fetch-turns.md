---
uid: 9e3a0206
id: neko-vrc-body.plugin.conversation.fetch-turns
parent: neko-vrc-body.plugin.conversation
name: {zh: "读取对话轮", en: "Fetch Conversation Turns"}
description:
  zh: >
      从 ctx.bus.conversations 读取最近的宿主对话轮，最新的在前。
      
  en: >
      Reading recent host conversation turns off ctx.bus.conversations, newest first.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.862Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 284
    end_line: 313
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._fetch_conversation_turns"
    description:
      zh: >
          从宿主总线读取最近的对话轮。
          
      en: >
          Read recent conversation turns from the host bus.
          
---
