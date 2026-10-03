---
uid: "9e430003"
id: neko-vrc-body.backend.nav-memory.store.sessions
parent: neko-vrc-body.backend.nav-memory.store
name: {zh: "会话生命周期", en: "Session Lifecycle"}
description:
  zh: >
      开会话、中断恢复与当前活跃会话状态；正在写的会话不许改动。
      
  en: >
      Opens sessions, recovers from interruption and reports the active session; a session being written must not be changed.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.642Z"
fingerprint: 8ac3ac2937f0825e7d07c1267ffa2e59000b81e0743552690708cb1b776bbb0e
source:
  - path: "backend/nav_memory.py"
    line: 278
    end_line: 348
apis:
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore.begin"
    description:
      zh: >
          开始一次新的导航会话。
          
      en: >
          Starts a new navigation session.
          
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore.recover"
    description:
      zh: >
          从磁盘恢复被中断的会话状态。
          
      en: >
          Recovers the state of an interrupted session from disk.
          
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore.end"
    description:
      zh: >
          结束并封存当前会话。
          
      en: >
          Ends and seals the current session.
          
---
