---
uid: 9e3c0108
id: neko-vrc-body.backend.client.errors
parent: neko-vrc-body.backend.client
name: {zh: "IPC 错误与远端 OSC 配置", en: "IPC Errors and Remote OSC Config"}
description:
  zh: >
      IPC 错误类型与远端 OSC 配置；区分「后端不可达」和「后端拒绝」。
      
  en: >
      IPC error types and the remote OSC configuration, separating "backend unreachable" from "backend rejected".
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.580Z"
fingerprint: 8f6d5a4ca8fb37bed50fe66446dfcb2eb54d67a9caca0125f881ac24d9844a7e
source:
  - path: "backend/client.py"
    line: 23
    end_line: 46
apis:
  - protocol: file
    path: "backend/client.py#BackendUnavailable"
    description:
      zh: >
          后端进程不可达或已退出的异常。
          
      en: >
          The exception raised when the backend process is unreachable or has exited.
          
  - protocol: file
    path: "backend/client.py#BackendRejected"
    description:
      zh: >
          后端可达但拒绝了本次请求的异常。
          
      en: >
          The exception raised when the backend is reachable but rejects the request.
          
---
