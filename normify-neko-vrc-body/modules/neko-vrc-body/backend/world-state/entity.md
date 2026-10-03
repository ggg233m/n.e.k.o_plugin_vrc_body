---
uid: 9e3f0016
id: neko-vrc-body.backend.world-state.entity
parent: neko-vrc-body.backend.world-state
name: {zh: "世界实体", en: "World Entity"}
description:
  zh: >
      一个有界的视觉世界实体假设，带来源、置信度与生命周期。
      
  en: >
      A bounded visual world-entity hypothesis carrying source, confidence, and lifecycle.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.774Z"
fingerprint: a086255eb138b205e2952faeb743fafefefaa2928a6212ffd04b152885bd28f3
source:
  - path: "backend/world_state.py"
    line: 188
    end_line: 268
apis:
  - protocol: file
    path: "backend/world_state.py#WorldEntity"
    description:
      zh: >
          世界实体假设的数据结构定义。
          
      en: >
          The data structure definition of a world-entity hypothesis.
          
  - protocol: file
    path: "backend/world_state.py#WorldEntity.from_mapping"
    description:
      zh: >
          从外部映射构造实体，忽略未知或非法字段。
          
      en: >
          Builds an entity from an external mapping, ignoring unknown or invalid fields.
          
  - protocol: file
    path: "backend/world_state.py#WorldEntity.expired"
    description:
      zh: >
          判断该实体是否已超过生命周期或 TTL。
          
      en: >
          Reports whether the entity has outlived its lifecycle or TTL.
          
---
