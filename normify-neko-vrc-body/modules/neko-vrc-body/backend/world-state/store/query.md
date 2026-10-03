---
uid: 9e3f001e
id: neko-vrc-body.backend.world-state.store.query
parent: neko-vrc-body.backend.world-state.store
name: {zh: "存储查询", en: "Store Query"}
description:
  zh: >
      对外读取：完整快照、修订日志与增量，均带新鲜度与不确定性标记。
      
  en: >
      Outward reads: full snapshot, revision journal, and delta, each carrying freshness and uncertainty markers.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.779Z"
fingerprint: a086255eb138b205e2952faeb743fafefefaa2928a6212ffd04b152885bd28f3
source:
  - path: "backend/world_state.py"
    line: 1000
    end_line: 1132
apis:
  - protocol: file
    path: "backend/world_state.py#WorldStateStore.snapshot"
    description:
      zh: >
          输出带新鲜度与不确定性标记的完整世界状态快照。
          
      en: >
          Emits the full world-state snapshot with freshness and uncertainty markers.
          
  - protocol: file
    path: "backend/world_state.py#WorldStateStore.delta"
    description:
      zh: >
          输出自指定修订号以来的增量变化。
          
      en: >
          Emits the incremental changes since a given revision number.
          
  - protocol: file
    path: "backend/world_state.py#WorldStateStore.journal"
    description:
      zh: >
          输出有界的修订日志。
          
      en: >
          Emits the bounded revision journal.
          
---
