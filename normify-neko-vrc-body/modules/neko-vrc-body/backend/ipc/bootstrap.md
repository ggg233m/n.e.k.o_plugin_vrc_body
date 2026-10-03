---
uid: 9e3c0107
id: neko-vrc-body.backend.ipc.bootstrap
parent: neko-vrc-body.backend.ipc
name: {zh: "后端进程引导", en: "Backend Process Bootstrap"}
description:
  zh: >
      后端进程引导：解析配置、准备 vendor 路径、绑定回环端口并进入服务循环。
      
  en: >
      Backend process bootstrap: parse the configuration, prepare vendor paths, bind the loopback port and enter the serving loop.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.601Z"
fingerprint: 6c2e840eb4584e5b87955eb2ea24baaf4194c875a3f3cb897255b978282779b3
source:
  - path: "backend/process.py"
    line: 537
    end_line: 668
apis:
  - protocol: file
    path: "backend/process.py#main"
    description:
      zh: >
          后端进程入口，负责启动引导与服务循环。
          
      en: >
          The backend process entry point that runs bootstrap and the serving loop.
          
---
