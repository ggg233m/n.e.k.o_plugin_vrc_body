---
uid: 9e3a0236
id: neko-vrc-body.plugin.tools.autonomy.goal
parent: neko-vrc-body.plugin.tools.autonomy
name: {zh: "自主目标", en: "Autonomy Goal"}
description:
  zh: >
      提交会话级自主目标：目标选择器加约束，且必须已显式授权。
      
  en: >
      Submitting a session-level autonomy goal: a target selector plus constraints, gated on explicit arm.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.881Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 3117
    end_line: 3230
apis:
  - protocol: rpc
    path: "vrc_autonomy_goal"
    description:
      zh: >
          工具：提交一个会话级自主目标。
          
      en: >
          Tool: submit a session-level autonomy goal.
          
---
