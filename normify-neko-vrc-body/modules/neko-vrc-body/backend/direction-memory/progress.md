---
uid: 9e3f0013
id: neko-vrc-body.backend.direction-memory.progress
parent: neko-vrc-body.backend.direction-memory
name: {zh: "进度积分", en: "Progress Integration"}
description:
  zh: >
      门控积分：只累计确实在直行的样本，并回报期间是否转向。
      
  en: >
      Gated integration: only samples that were genuinely moving straight are accumulated, and whether a turn occurred meanwhile is reported.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.595Z"
fingerprint: 10cac7e571fabcdb167f981ba4b09c03bf51d330c67e6c7e3f92b5caac9a3eee
source:
  - path: "backend/direction_memory.py"
    line: 456
    end_line: 489
apis:
  - protocol: file
    path: "backend/direction_memory.py#integrate_progress"
    description:
      zh: >
          按门控规则累计直行样本并回报期间是否转向。
          
      en: >
          Accumulates straight-moving samples under the gating rule and reports whether a turn happened meanwhile.
          
---
