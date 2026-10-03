---
uid: 9e3f0020
id: neko-vrc-body.backend.world-model.core
parent: neko-vrc-body.backend.world-model
name: {zh: "世界模型骨架", en: "World Model Core"}
description:
  zh: >
      线程安全的世界身份与启停骨架；世界标识只靠手动设置。
      
  en: >
      A thread-safe world identity and start/stop skeleton whose world identifier is only set manually.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.773Z"
fingerprint: ad05cd240b621788800bf26ba2b74882ff02bf5eb22a05e88865a74cc72bee0c
source:
  - path: "backend/world_model.py"
    line: 58
    end_line: 264
apis:
  - protocol: file
    path: "backend/world_model.py#WorldModel.set_world"
    description:
      zh: >
          手动设置当前世界标识。
          
      en: >
          Manually sets the current world identifier.
          
  - protocol: file
    path: "backend/world_model.py#WorldModel.identity"
    description:
      zh: >
          返回当前线程安全的世界身份。
          
      en: >
          Returns the current thread-safe world identity.
          
  - protocol: file
    path: "backend/world_model.py#WorldModel.memory_partition"
    description:
      zh: >
          给出该世界的记忆分区键。
          
      en: >
          Returns the memory partition key for this world.
          
  - protocol: file
    path: "backend/world_model.py#WorldModel.start"
    description:
      zh: >
          启动世界模型骨架。
          
      en: >
          Starts the world model skeleton.
          
---
