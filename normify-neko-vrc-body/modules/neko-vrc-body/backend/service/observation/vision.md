---
uid: 9e3e0020
id: neko-vrc-body.backend.service.observation.vision
parent: neko-vrc-body.backend.service.observation
name: {zh: "视觉观测处理", en: "Vision Observation Handling"}
description:
  zh: >
      把检测器输出并入世界状态存储，以及检查待处理的语义选择器是否已被满足。
      
  en: >
      Folding detector output into the world store, and checking whether a pending semantic selector is already satisfied.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.746Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 3140
    end_line: 3218
apis:
  - protocol: file
    path: "backend/service.py#BackendService._on_vision_observation"
    description:
      zh: >
          把一次视觉观测并入世界状态存储。
          
      en: >
          Fold one vision observation into the world store.
          
  - protocol: file
    path: "backend/service.py#BackendService._semantic_selector_is_satisfied"
    description:
      zh: >
          判定某个选择器是否已被当前世界内容满足。
          
      en: >
          Decide whether a selector is satisfied by what the world currently holds.
          
deps:
  - kind: dataflow
    to: neko-vrc-body.backend.world-state.store.ingest
    label: {zh: "摄入该批次", en: "Ingest the batch"}
  - kind: call
    to: neko-vrc-body.backend.world-state.store.query
    label: {zh: "回读实体", en: "Read entities back"}
---
