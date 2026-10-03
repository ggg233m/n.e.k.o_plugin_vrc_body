---
uid: 9e3c0102
id: neko-vrc-body.backend.ipc.http-handler.auth
parent: neko-vrc-body.backend.ipc.http-handler
name: {zh: "鉴权与日志静默", en: "Authorization and Log Silence"}
description:
  zh: >
      请求鉴权与日志静默；未授权请求在进入任何路由前被拒绝。
      
  en: >
      Request authorization and log silencing; unauthorized requests are rejected before reaching any route.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.602Z"
fingerprint: 6c2e840eb4584e5b87955eb2ea24baaf4194c875a3f3cb897255b978282779b3
source:
  - path: "backend/process.py"
    line: 127
    end_line: 137
apis:
  - protocol: file
    path: "backend/process.py#BackendRequestHandler._authorized"
    description:
      zh: >
          校验请求的鉴权凭据，失败即拒绝。
          
      en: >
          Validates request credentials and rejects on failure.
          
---
