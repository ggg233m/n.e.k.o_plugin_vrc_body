---
uid: 9e3f001b
id: neko-vrc-body.backend.world-state.store.watermark
parent: neko-vrc-body.backend.world-state.store
name: {zh: "存储水位线", en: "Store Watermark"}
description:
  zh: >
      生命周期水位线：来源重置时压住旧实体，而不是让它们继续参与判断。
      
  en: >
      Lifecycle watermarks: a source reset suppresses stale entities instead of letting them keep influencing decisions.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.780Z"
fingerprint: a086255eb138b205e2952faeb743fafefefaa2928a6212ffd04b152885bd28f3
source:
  - path: "backend/world_state.py"
    line: 533
    end_line: 641
apis:
  - protocol: file
    path: "backend/world_state.py#WorldStateStore._watermark_blocks"
    description:
      zh: >
          判断某条水位线是否应当压住对应来源的旧实体。
          
      en: >
          Decides whether a watermark should suppress the stale entities of its source.
          
  - protocol: file
    path: "backend/world_state.py#WorldStateStore._prune_lifecycle_watermarks"
    description:
      zh: >
          裁剪已失效的水位线，避免它们无限增长。
          
      en: >
          Prunes no-longer-valid watermarks so they do not grow without bound.
          
---
