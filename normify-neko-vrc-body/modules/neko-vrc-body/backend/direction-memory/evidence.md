---
uid: 9e3f0011
id: neko-vrc-body.backend.direction-memory.evidence
parent: neko-vrc-body.backend.direction-memory
name: {zh: "方向证据", en: "Direction Evidence"}
description:
  zh: >
      一个方向扇区上的实测证据与模型偏好分开存放，以及一段移动结束后交回的实测事实。
      
  en: >
      Measured evidence on a direction sector kept separate from model preference, plus the measured facts handed back after a movement segment.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.594Z"
fingerprint: 10cac7e571fabcdb167f981ba4b09c03bf51d330c67e6c7e3f92b5caac9a3eee
source:
  - path: "backend/direction_memory.py"
    line: 127
    end_line: 218
apis:
  - protocol: file
    path: "backend/direction_memory.py#DirectionEvidence"
    description:
      zh: >
          单个方向扇区上的证据结构，实测值与模型偏好分开记录。
          
      en: >
          The evidence structure for one direction sector, recording measured values separately from model preference.
          
  - protocol: file
    path: "backend/direction_memory.py#SegmentOutcome"
    description:
      zh: >
          一段移动结束后交回的实测事实。
          
      en: >
          The measured facts returned after a movement segment finishes.
          
---
