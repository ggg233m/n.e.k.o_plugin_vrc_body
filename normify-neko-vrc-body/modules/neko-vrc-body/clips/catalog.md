---
uid: 9e3a001d
id: neko-vrc-body.clips.catalog
parent: neko-vrc-body.clips
name: {zh: "动作目录", en: "Motion Catalog"}
description:
  zh: >
      加载 motions/catalog.json，并在全程不解析 .nya 负载的前提下挑选片段。
      
  en: >
      Loading motions/catalog.json and selecting clips without ever parsing a .nya payload.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.804Z"
fingerprint: 4f5137f759f8da94add80ee10e6ee7eef243e06cd44aab40d60a846e61b26983
source:
  - path: "motion_catalog.py"
    line: 14
    end_line: 199
apis:
  - protocol: file
    path: "motion_catalog.py#MotionMetadata"
    description:
      zh: >
          一条已编目片段的校验后元数据。
          
      en: >
          Validated metadata for one catalogued clip.
          
  - protocol: file
    path: "motion_catalog.py#MotionCatalog.select"
    description:
      zh: >
          为一个语义意图确定性地挑选片段。
          
      en: >
          Select a clip deterministically for a semantic intent.
          
  - protocol: file
    path: "motion_catalog.py#MotionCatalog.summary"
    description:
      zh: >
          为状态回读汇总目录。
          
      en: >
          Summarise the catalogue for the status read-out.
          
---
