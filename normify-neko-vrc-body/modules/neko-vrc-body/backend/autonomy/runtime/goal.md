---
uid: 9e3f000b
id: neko-vrc-body.backend.autonomy.runtime.goal
parent: neko-vrc-body.backend.autonomy.runtime
name: {zh: "运行时目标提交", en: "Runtime Goal Submission"}
description:
  zh: >
      目标提交：校验种类、选择器与约束，并拒绝未授权的会话。
      
  en: >
      Goal submission: validates kind, selector, and constraints, and rejects unauthorized sessions.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.573Z"
fingerprint: 956f89ee80cd18326626bcfed51b616a94c813b4ec5dd9fa58499c2a52baf96a
source:
  - path: "backend/autonomy.py"
    line: 258
    end_line: 330
apis:
  - protocol: file
    path: "backend/autonomy.py#AutonomyRuntime.submit_goal"
    description:
      zh: >
          提交一个自主目标，校验种类、选择器与约束，并拒绝未授权的会话。
          
      en: >
          Submits an autonomy goal, validating kind, selector, and constraints and rejecting unauthorized sessions.
          
---
