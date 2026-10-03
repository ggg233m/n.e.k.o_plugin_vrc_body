---
uid: 9e3a0207
id: neko-vrc-body.plugin.conversation.disk-records
parent: neko-vrc-body.plugin.conversation
name: {zh: "磁盘对话记录", en: "On-disk Chat Records"}
description:
  zh: >
      回退路径：从宿主的磁盘对话记录重建对话轮。
      
  en: >
      Fallback that reconstructs conversation turns from the host's on-disk chat records.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.860Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 315
    end_line: 368
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._read_disk_chat_records"
    description:
      zh: >
          读取宿主持久化到磁盘的对话记录。
          
      en: >
          Read conversation records the host persisted on disk.
          
---
