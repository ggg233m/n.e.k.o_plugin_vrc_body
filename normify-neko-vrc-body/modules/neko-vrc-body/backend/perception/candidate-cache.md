---
uid: "9e400014"
id: neko-vrc-body.backend.perception.candidate-cache
parent: neko-vrc-body.backend.perception
name: {zh: "语义候选缓存", en: "Semantic Candidate Cache"}
description:
  zh: >
      只驻留内存的语义候选与外观原型，带修剪与容量上限。
      
  en: >
      An in-memory-only cache of semantic candidates and appearance prototypes, with pruning and a capacity bound.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.693Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 1937
    end_line: 2216
apis:
  - protocol: file
    path: "backend/vision.py#SemanticCandidateCache.bind"
    description:
      zh: >
          把缓存绑定到当前运行时与目标标识上，开启一次会话。
          
      en: >
          Binds the cache to the current runtime and target identity, opening a session.
          
  - protocol: file
    path: "backend/vision.py#SemanticCandidateCache.enrich_observation"
    description:
      zh: >
          用候选与外观原型补全观测中的语义字段与置信度。
          
      en: >
          Enriches the observation's semantic fields and confidence using candidates and appearance prototypes.
          
  - protocol: file
    path: "backend/vision.py#SemanticCandidateCache.snapshot"
    description:
      zh: >
          导出当前候选与原型集合的只读快照。
          
      en: >
          Exports a read-only snapshot of the current candidate and prototype set.
          
---
