---
uid: 9e3f000a
id: neko-vrc-body.backend.autonomy.runtime.authorization
parent: neko-vrc-body.backend.autonomy.runtime
name: {zh: "运行时授权", en: "Runtime Authorization"}
description:
  zh: >
      手动授权、会到期时自动失效的授权状态机。
      
  en: >
      An authorization state machine that is armed manually and expires automatically.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.573Z"
fingerprint: 956f89ee80cd18326626bcfed51b616a94c813b4ec5dd9fa58499c2a52baf96a
source:
  - path: "backend/autonomy.py"
    line: 183
    end_line: 256
apis:
  - protocol: file
    path: "backend/autonomy.py#AutonomyRuntime.arm"
    description:
      zh: >
          手动授权本次会话的自主运行时，并设置到期时间。
          
      en: >
          Manually arms the autonomy runtime for this session and sets its expiry.
          
  - protocol: file
    path: "backend/autonomy.py#AutonomyRuntime.disarm"
    description:
      zh: >
          撤销授权，使后续目标提交被拒绝。
          
      en: >
          Revokes the authorization so later goal submissions are rejected.
          
---
