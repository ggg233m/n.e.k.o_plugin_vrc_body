---
uid: 9e3f001f
id: neko-vrc-body.backend.world-model.identity
parent: neko-vrc-body.backend.world-model
name: {zh: "世界标识", en: "World Identity"}
description:
  zh: >
      把任意 world_key 规整成安全的单段目录名，navmesh 记忆分区也用它。
      
  en: >
      Normalizes any world_key into a safe single-segment directory name, which navmesh memory partitions also use.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.773Z"
fingerprint: ad05cd240b621788800bf26ba2b74882ff02bf5eb22a05e88865a74cc72bee0c
source:
  - path: "backend/world_model.py"
    line: 43
    end_line: 55
apis:
  - protocol: file
    path: "backend/world_model.py#sanitize_world_key"
    description:
      zh: >
          把任意 world_key 规整为安全的单段目录名。
          
      en: >
          Normalizes an arbitrary world_key into a safe single-segment directory name.
          
---
