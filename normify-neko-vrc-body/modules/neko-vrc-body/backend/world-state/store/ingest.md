---
uid: 9e3f001c
id: neko-vrc-body.backend.world-state.store.ingest
parent: neko-vrc-body.backend.world-state.store
name: {zh: "存储写入", en: "Store Ingest"}
description:
  zh: >
      合并一次观测批次，并只把真正变化的实体写进修订日志。
      
  en: >
      Merges one observation batch and journals only the entities that genuinely changed.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.777Z"
fingerprint: a086255eb138b205e2952faeb743fafefefaa2928a6212ffd04b152885bd28f3
source:
  - path: "backend/world_state.py"
    line: 643
    end_line: 814
apis:
  - protocol: file
    path: "backend/world_state.py#WorldStateStore.ingest"
    description:
      zh: >
          合并一次观测批次，并只把真正变化的实体写进修订日志。
          
      en: >
          Merges one observation batch and journals only the entities that genuinely changed.
          
---
