---
uid: 9e3e000f
id: neko-vrc-body.backend.service.autonomy.goal
parent: neko-vrc-body.backend.service.autonomy
name: {zh: "目标提交", en: "Goal Submission"}
description:
  zh: >
      目标提交：选择器校验、仅人形可检测的限制，以及对记忆已排除方位的拒绝。
      
  en: >
      Goal submission: selector validation, humanoid-only detectability and the refusal of bearings the memory already ruled out.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.727Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 1125
    end_line: 1295
apis:
  - protocol: file
    path: "backend/service.py#BackendService.autonomy_goal"
    description:
      zh: >
          提交一个会话级自主目标。
          
      en: >
          Submit a session-level autonomy goal.
          
  - protocol: file
    path: "backend/service.py#BackendService._refuse_blocked_bearing"
    description:
      zh: >
          拒绝方向记忆判定为受阻的方位。
          
      en: >
          Refuse a bearing the direction memory says is blocked.
          
---
