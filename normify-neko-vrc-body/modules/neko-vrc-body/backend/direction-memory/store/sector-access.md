---
uid: "9e500022"
id: neko-vrc-body.backend.direction-memory.store.sector-access
parent: neko-vrc-body.backend.direction-memory.store
name: {zh: "扇区访问", en: "Sector Access"}
description:
  zh: >
      扇区条目的获取与记忆扇区、相对方位的换算。
      
  en: >
      Access to sector entries and the mapping to memory sectors and relative bearings.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.597Z"
fingerprint: 10cac7e571fabcdb167f981ba4b09c03bf51d330c67e6c7e3f92b5caac9a3eee
source:
  - path: "backend/direction_memory.py"
    line: 222
    end_line: 255
apis:
  - protocol: file
    path: "backend/direction_memory.py#_entry"
    description:
      zh: >
          返回某个扇区的记忆条目。
          
      en: >
          Returns the memory entry of a sector.
          
  - protocol: file
    path: "backend/direction_memory.py#_memory_sector"
    description:
      zh: >
          把方位映射到记忆扇区。
          
      en: >
          Maps a bearing onto a memory sector.
          
  - protocol: file
    path: "backend/direction_memory.py#_relative_bearing"
    description:
      zh: >
          把世界方位换算为相对方位。
          
      en: >
          Converts a world bearing to a relative bearing.
          
---
