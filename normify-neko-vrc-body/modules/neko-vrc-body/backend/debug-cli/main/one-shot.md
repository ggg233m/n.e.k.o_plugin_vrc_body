---
uid: "9e500009"
id: neko-vrc-body.backend.debug-cli.main.one-shot
parent: neko-vrc-body.backend.debug-cli.main
name: {zh: "一次性分发", en: "One-Shot Dispatch"}
description:
  zh: >
      单条命令的一次性请求分发。
      
  en: >
      One-shot request dispatch for a single command.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.593Z"
fingerprint: ea1de34602a67e7be6b4c263a343c3d70a69e31f189f484adadc872b211d42a8
source:
  - path: "backend/debug_cli.py"
    line: 169
    end_line: 260
apis:
  - protocol: file
    path: "backend/debug_cli.py#_run_once"
    description:
      zh: >
          分发单条命令并发出一次请求。
          
      en: >
          Dispatches a single command and issues one request.
          
---
