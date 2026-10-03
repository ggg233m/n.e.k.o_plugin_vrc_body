---
uid: 9e3f0005
id: neko-vrc-body.backend.cognition.planning
parent: neko-vrc-body.backend.cognition
name: {zh: "规划", en: "Planning"}
description:
  zh: >
      确定性的基线规划器，把高层计划归一化为严格 JSON。
      
  en: >
      A deterministic baseline planner that normalizes high-level plans into strict JSON.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.587Z"
fingerprint: ef89d358bc9669d29d55dc2f3e13d67ab8ffb2d1c9f55ccdf7de4175419d1d8a
source:
  - path: "backend/cognition.py"
    line: 551
    end_line: 706
apis:
  - protocol: file
    path: "backend/cognition.py#SkillPlanner.plan"
    description:
      zh: >
          把高层意图展开为归一化的计划步骤序列。
          
      en: >
          Expands a high-level intent into a normalized sequence of plan steps.
          
  - protocol: file
    path: "backend/cognition.py#Plan"
    description:
      zh: >
          完整计划结构：意图、步骤序列与归一化结果。
          
      en: >
          The full plan structure: intent, step sequence, and normalization outcome.
          
---
