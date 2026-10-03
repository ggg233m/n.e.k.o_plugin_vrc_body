---
uid: 9e3f0004
id: neko-vrc-body.backend.cognition.observation
parent: neko-vrc-body.backend.cognition
name: {zh: "观测与状态估计", en: "Observation and State Estimation"}
description:
  zh: >
      带时间戳的观测记录，以及融合新鲜度与置信度的状态估计器。
      
  en: >
      Timestamped observation records plus a state estimator that fuses freshness with confidence.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.587Z"
fingerprint: ef89d358bc9669d29d55dc2f3e13d67ab8ffb2d1c9f55ccdf7de4175419d1d8a
source:
  - path: "backend/cognition.py"
    line: 442
    end_line: 547
apis:
  - protocol: file
    path: "backend/cognition.py#StateEstimator.ingest"
    description:
      zh: >
          摄入一条观测并刷新有界的估计状态。
          
      en: >
          Ingests one observation and refreshes the bounded estimated state.
          
  - protocol: file
    path: "backend/cognition.py#StateEstimator.snapshot"
    description:
      zh: >
          输出同时带新鲜度与置信度的状态快照。
          
      en: >
          Emits a state snapshot that carries both freshness and confidence.
          
  - protocol: file
    path: "backend/cognition.py#ObservationRecord"
    description:
      zh: >
          单条带时间戳的观测记录结构。
          
      en: >
          The structure of one timestamped observation record.
          
---
