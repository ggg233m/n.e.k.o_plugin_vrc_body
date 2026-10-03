---
uid: 9e3e0021
id: neko-vrc-body.backend.service.observation.world
parent: neko-vrc-body.backend.service.observation
name: {zh: "世界注入与规划", en: "World Ingestion and Planning"}
description:
  zh: >
      外部世界注入、确定性规划，以及闭合认知循环的反馈。
      
  en: >
      External world ingestion, deterministic planning and the feedback that closes the cognition loop.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.747Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 3220
    end_line: 3250
apis:
  - protocol: file
    path: "backend/service.py#BackendService.ingest_world"
    description:
      zh: >
          接收由外部适配器提供的世界快照。
          
      en: >
          Ingest a world snapshot supplied by an external adapter.
          
  - protocol: file
    path: "backend/service.py#BackendService.plan"
    description:
      zh: >
          为某个目标产出严格 JSON 的计划。
          
      en: >
          Produce a strict-JSON plan for a goal.
          
  - protocol: file
    path: "backend/service.py#BackendService.cognition_feedback"
    description:
      zh: >
          把上一次计划的结果记入认知循环。
          
      en: >
          Record the outcome of the last plan into the cognition loop.
          
---
