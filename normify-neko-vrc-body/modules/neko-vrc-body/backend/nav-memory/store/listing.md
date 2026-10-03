---
uid: "9e430004"
id: neko-vrc-body.backend.nav-memory.store.listing
parent: neko-vrc-body.backend.nav-memory.store
name: {zh: "记忆列举与查询", en: "Listing & Query"}
description:
  zh: >
      列举世界与会话、读取详情与缩略图，以及改标签与钉住。
      
  en: >
      Lists worlds and sessions, reads details and thumbnails, and edits labels and pinning.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.641Z"
fingerprint: 8ac3ac2937f0825e7d07c1267ffa2e59000b81e0743552690708cb1b776bbb0e
source:
  - path: "backend/nav_memory.py"
    line: 351
    end_line: 410
apis:
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore.list_worlds"
    description:
      zh: >
          列出所有已记录记忆的世界。
          
      en: >
          Lists every world that has recorded memories.
          
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore.list_sessions"
    description:
      zh: >
          列出某个世界下的历史会话。
          
      en: >
          Lists the past sessions stored under one world.
          
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore.update_session"
    description:
      zh: >
          修改会话的标签与钉住状态。
          
      en: >
          Updates a session's labels and pinned state.
          
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore.thumbnail"
    description:
      zh: >
          读取某个会话的缩略图。
          
      en: >
          Reads a session thumbnail.
          
---
