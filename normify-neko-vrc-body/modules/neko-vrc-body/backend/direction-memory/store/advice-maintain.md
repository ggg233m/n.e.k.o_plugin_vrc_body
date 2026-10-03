---
uid: "9e700003"
id: neko-vrc-body.backend.direction-memory.store.advice-maintain
parent: neko-vrc-body.backend.direction-memory.store
name: {zh: "建议与维护", en: "Advice and Maintenance"}
description:
  zh: >
      记录结果、渲染建议，以及让记忆保持有界的过期与重置。
      
  en: >
      Recording outcomes, rendering advice, and the expiry and reset that keep the memory bounded.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.595Z"
fingerprint: 10cac7e571fabcdb167f981ba4b09c03bf51d330c67e6c7e3f92b5caac9a3eee
source:
  - path: "backend/direction_memory.py"
    line: 384
    end_line: 453
apis:
  - protocol: file
    path: "backend/direction_memory.py#DirectionMemory.record"
    description:
      zh: >
          记录一段移动结果。
          
      en: >
          Record one segment outcome.
          
  - protocol: file
    path: "backend/direction_memory.py#DirectionMemory.advice"
    description:
      zh: >
          为当前朝向渲染方向建议。
          
      en: >
          Render direction advice for the current heading.
          
  - protocol: file
    path: "backend/direction_memory.py#DirectionMemory._prune"
    description:
      zh: >
          丢弃已过期的条目。
          
      en: >
          Drop entries that have expired.
          
---
