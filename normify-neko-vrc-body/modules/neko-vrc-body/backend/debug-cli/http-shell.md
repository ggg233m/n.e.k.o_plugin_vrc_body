---
uid: 9e3e0023
id: neko-vrc-body.backend.debug-cli.http-shell
parent: neko-vrc-body.backend.debug-cli
name: {zh: "持久 HTTP Shell", en: "Persistent HTTP Shell"}
description:
  zh: >
      为高频调用方复用同一条回环连接的持久 JSON-lines shell。
      
  en: >
      A persistent JSON-lines shell that reuses one loopback connection for high-frequency callers.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.591Z"
fingerprint: ea1de34602a67e7be6b4c263a343c3d70a69e31f189f484adadc872b211d42a8
source:
  - path: "backend/debug_cli.py"
    line: 39
    end_line: 110
apis:
  - protocol: file
    path: "backend/debug_cli.py#_PersistentHttpClient"
    description:
      zh: >
          复用同一条回环连接的 JSON-lines shell 传输层。
          
      en: >
          A JSON-lines shell transport that reuses one loopback connection.
          
  - protocol: file
    path: "backend/debug_cli.py#_run_shell"
    description:
      zh: >
          运行一个持久的 JSON-lines 控制会话。
          
      en: >
          Run a persistent JSON-lines control session.
          
---
