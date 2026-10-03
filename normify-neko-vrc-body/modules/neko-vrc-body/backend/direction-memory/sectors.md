---
uid: 9e3f0010
id: neko-vrc-body.backend.direction-memory.sectors
parent: neko-vrc-body.backend.direction-memory
name: {zh: "方向扇区", en: "Direction Sectors"}
description:
  zh: >
      把任意角度折进约定区间，并映射到扇区索引与代表方向。
      
  en: >
      Folds any angle into the agreed interval and maps it to a sector index and representative direction.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.595Z"
fingerprint: 10cac7e571fabcdb167f981ba4b09c03bf51d330c67e6c7e3f92b5caac9a3eee
source:
  - path: "backend/direction_memory.py"
    line: 39
    end_line: 123
apis:
  - protocol: file
    path: "backend/direction_memory.py#normalize_bearing"
    description:
      zh: >
          把任意角度折进约定的度数区间。
          
      en: >
          Folds an arbitrary angle into the agreed degree interval.
          
  - protocol: file
    path: "backend/direction_memory.py#sector_of"
    description:
      zh: >
          把方位角映射为离散的扇区索引。
          
      en: >
          Maps a bearing to a discrete sector index.
          
  - protocol: file
    path: "backend/direction_memory.py#sector_center_deg"
    description:
      zh: >
          给出某个扇区的代表中心方向。
          
      en: >
          Returns the representative center direction of a sector.
          
---
