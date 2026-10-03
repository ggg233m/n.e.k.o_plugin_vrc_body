---
uid: 9e3f0014
id: neko-vrc-body.backend.world-state.ids
parent: neko-vrc-body.backend.world-state
name: {zh: "实体标识", en: "Entity Identity"}
description:
  zh: >
      实体 ID 生成：不受类别抖动影响的跨帧 ID，以及世界日志与视觉适配器共用的玩家 ID。
      
  en: >
      Entity ID generation: cross-frame IDs immune to category jitter, plus the player ID shared by the world log and the vision adapter.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.775Z"
fingerprint: a086255eb138b205e2952faeb743fafefefaa2928a6212ffd04b152885bd28f3
source:
  - path: "backend/world_state.py"
    line: 19
    end_line: 58
  - path: "backend/world_state.py"
    line: 83
    end_line: 184
apis:
  - protocol: file
    path: "backend/world_state.py#stable_track_entity_id"
    description:
      zh: >
          由跟踪轨迹生成跨帧稳定且不受类别抖动影响的实体 ID。
          
      en: >
          Builds a cross-frame stable entity ID from a track that is immune to category jitter.
          
  - protocol: file
    path: "backend/world_state.py#vrchat_player_entity_id"
    description:
      zh: >
          生成世界日志与视觉适配器共用的玩家实体 ID。
          
      en: >
          Builds the player entity ID shared by the world log and the vision adapter.
          
  - protocol: file
    path: "backend/world_state.py#stable_entity_id"
    description:
      zh: >
          由任意来源字段组合出确定的实体 ID。
          
      en: >
          Derives a deterministic entity ID from arbitrary source fields.
          
---
