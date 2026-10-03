---
uid: "9e400020"
id: neko-vrc-body.backend.perception.runtime.ingest
parent: neko-vrc-body.backend.perception.runtime
name: {zh: "观测注入与快照", en: "Observation Ingest and Snapshot"}
description:
  zh: >
      外部观测注入、整帧处理与对外快照/增量读取。
      
  en: >
      External observation ingestion, whole-frame processing, and outward snapshot/delta reads.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.709Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 3906
    end_line: 4181
apis:
  - protocol: file
    path: "backend/vision.py#VisionRuntime.ingest"
    description:
      zh: >
          接收外部注入的观测并纳入运行时状态。
          
      en: >
          Accepts an externally injected observation and folds it into the runtime state.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime.process_frame"
    description:
      zh: >
          对一整帧做检测、语义与缓存处理的同步入口。
          
      en: >
          The synchronous entry point that runs detection, semantics, and caching over a whole frame.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime.snapshot"
    description:
      zh: >
          返回当前完整观测快照。
          
      en: >
          Returns the current full observation snapshot.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime.delta"
    description:
      zh: >
          返回自上次读取以来的观测增量。
          
      en: >
          Returns the observation delta since the last read.
          
---
