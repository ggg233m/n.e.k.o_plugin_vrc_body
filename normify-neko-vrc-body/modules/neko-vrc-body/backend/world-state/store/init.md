---
uid: 9e3f0019
id: neko-vrc-body.backend.world-state.store.init
parent: neko-vrc-body.backend.world-state.store
name: {zh: "存储初始化", en: "Store Init"}
description:
  zh: >
      有界线程安全存储的构造：容量、TTL、水位线与持久化开关。
      
  en: >
      Construction of the bounded thread-safe store: capacity, TTL, watermarks, and persistence switches.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.778Z"
fingerprint: a086255eb138b205e2952faeb743fafefefaa2928a6212ffd04b152885bd28f3
source:
  - path: "backend/world_state.py"
    line: 313
    end_line: 376
apis:
  - protocol: file
    path: "backend/world_state.py#WorldStateStore.__init__"
    description:
      zh: >
          构造有界线程安全存储所需的容量、TTL、水位线与持久化开关。
          
      en: >
          Configures the capacity, TTL, watermarks, and persistence switches of the bounded thread-safe store.
          
---
