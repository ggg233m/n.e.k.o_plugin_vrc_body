---
uid: 9e3f001a
id: neko-vrc-body.backend.world-state.store.persistence
parent: neko-vrc-body.backend.world-state.store
name: {zh: "存储持久化", en: "Store Persistence"}
description:
  zh: >
      只持久化可复用世界事实，并维护有界的修订日志。
      
  en: >
      Persists only reusable world facts and maintains a bounded revision journal.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.779Z"
fingerprint: a086255eb138b205e2952faeb743fafefefaa2928a6212ffd04b152885bd28f3
source:
  - path: "backend/world_state.py"
    line: 378
    end_line: 531
apis:
  - protocol: file
    path: "backend/world_state.py#WorldStateStore._persist_locked"
    description:
      zh: >
          在持锁状态下把可复用世界事实写入持久层。
          
      en: >
          Writes reusable world facts to the persistence layer while holding the lock.
          
  - protocol: file
    path: "backend/world_state.py#WorldStateStore._load_persisted"
    description:
      zh: >
          从持久层恢复此前的世界状态。
          
      en: >
          Restores the previously persisted world state from the persistence layer.
          
---
