---
uid: "9e500004"
id: neko-vrc-body.backend.navigator.config.target-observation
parent: neko-vrc-body.backend.navigator.config
name: {zh: "目标观测缓存", en: "Target Observation Cache"}
description:
  zh: >
      目标观测缓存与检测器无关的空间提示读取。
      
  en: >
      Cached target observations and detector-independent spatial-hint reads.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.667Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 273
    end_line: 292
  - path: "backend/navigator.py"
    line: 303
    end_line: 384
apis:
  - protocol: file
    path: "backend/navigator.py#_spatial_hint"
    description:
      zh: >
          读取与检测器无关的空间提示。
          
      en: >
          Reads a detector-independent spatial hint.
          
---
