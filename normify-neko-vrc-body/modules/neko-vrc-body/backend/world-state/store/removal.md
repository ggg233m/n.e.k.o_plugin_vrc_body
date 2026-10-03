---
uid: 9e3f001d
id: neko-vrc-body.backend.world-state.store.removal
parent: neko-vrc-body.backend.world-state.store
name: {zh: "存储移除", en: "Store Removal"}
description:
  zh: >
      显式移除实体或整个来源，而不是靠超时悄悄清理。
      
  en: >
      Explicitly removes entities or a whole source instead of quietly cleaning them up by timeout.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.780Z"
fingerprint: a086255eb138b205e2952faeb743fafefefaa2928a6212ffd04b152885bd28f3
source:
  - path: "backend/world_state.py"
    line: 816
    end_line: 998
apis:
  - protocol: file
    path: "backend/world_state.py#WorldStateStore.remove_entity"
    description:
      zh: >
          按 ID 显式移除单个实体。
          
      en: >
          Explicitly removes a single entity by ID.
          
  - protocol: file
    path: "backend/world_state.py#WorldStateStore.remove_entities_by_source"
    description:
      zh: >
          显式移除某个来源下的全部实体。
          
      en: >
          Explicitly removes every entity belonging to one source.
          
---
