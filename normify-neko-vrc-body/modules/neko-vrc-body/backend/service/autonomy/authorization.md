---
uid: 9e3e000e
id: neko-vrc-body.backend.service.autonomy.authorization
parent: neko-vrc-body.backend.service.autonomy
name: {zh: "授权状态", en: "Authorization State"}
description:
  zh: >
      授权状态本身：读取、授权、撤销，以及清空与目标绑定的待处理意图。
      
  en: >
      The authorization state itself: read, arm, disarm, and clearing the pending intent that was bound to the goal.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.727Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 1090
    end_line: 1123
apis:
  - protocol: file
    path: "backend/service.py#BackendService.autonomy_snapshot"
    description:
      zh: >
          读取当前自主授权与目标状态。
          
      en: >
          Read the current autonomy authorization and goal state.
          
  - protocol: file
    path: "backend/service.py#BackendService.autonomy_arm"
    description:
      zh: >
          授予自主操作授权。
          
      en: >
          Arm autonomy authorization.
          
  - protocol: file
    path: "backend/service.py#BackendService.autonomy_disarm"
    description:
      zh: >
          撤销自主操作授权。
          
      en: >
          Withdraw autonomy authorization.
          
  - protocol: file
    path: "backend/service.py#BackendService._clear_pending_semantic_intent"
    description:
      zh: >
          丢弃与目标绑定的待处理语义意图。
          
      en: >
          Drop any pending semantic intent tied to the goal.
          
---
