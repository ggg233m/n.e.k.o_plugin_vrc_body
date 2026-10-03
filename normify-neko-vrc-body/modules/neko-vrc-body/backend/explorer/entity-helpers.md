---
uid: 9e3f000d
id: neko-vrc-body.backend.explorer.entity-helpers
parent: neko-vrc-body.backend.explorer
name: {zh: "实体字段读取助手", en: "Entity Field Readers"}
description:
  zh: >
      与检测器无关的实体字段读取：语义类别与方位角。
      
  en: >
      Detector-agnostic entity field reads: semantic category and bearing.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.598Z"
fingerprint: 3ceb238589b7465b0375c659e6db1f5f984144bfe544a81d29d28423b1db0ad9
source:
  - path: "backend/explorer.py"
    line: 15
    end_line: 52
apis:
  - protocol: file
    path: "backend/explorer.py#_bearing"
    description:
      zh: >
          从实体位置算出相对观察者或原点的方位角。
          
      en: >
          Computes an entity's bearing relative to the observer or origin.
          
  - protocol: file
    path: "backend/explorer.py#_semantic_type"
    description:
      zh: >
          读取并归一化实体的语义类别。
          
      en: >
          Reads and normalizes the semantic category of an entity.
          
---
