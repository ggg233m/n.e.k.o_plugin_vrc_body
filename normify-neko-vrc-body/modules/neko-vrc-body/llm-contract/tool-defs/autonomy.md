---
uid: 9e3a0105
id: neko-vrc-body.llm-contract.tool-defs.autonomy
parent: neko-vrc-body.llm-contract.tool-defs
name: {zh: "自主工具", en: "Autonomy Tools"}
description:
  zh: >
      自主目标提交与一步有界闲逛的 Schema。
      
  en: >
      Schemas for autonomy goal submission and one bounded wander step.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.836Z"
fingerprint: 348077cad7b4ebac68be08404f65a6094ebab38baa302e5e51c5b85edc242009
source:
  - path: "tool_defs.py"
    line: 161
    end_line: 296
apis:
  - protocol: file
    path: "tool_defs.py#VRC_AUTONOMY_GOAL"
    description:
      zh: >
          工具 Schema：提交一个会话级自主目标。
          
      en: >
          Tool schema: submit a session-level autonomy goal.
          
  - protocol: file
    path: "tool_defs.py#VRC_WANDER_STEP"
    description:
      zh: >
          工具 Schema：推进一步有界的自由漫游。
          
      en: >
          Tool schema: advance one bounded free-roam step.
          
---
