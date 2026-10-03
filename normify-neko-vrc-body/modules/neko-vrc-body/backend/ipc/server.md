---
uid: 9e3c0106
id: neko-vrc-body.backend.ipc.server
parent: neko-vrc-body.backend.ipc
name: {zh: "回环服务器", en: "Loopback Server"}
description:
  zh: >
      多线程回环服务器与配置文件读取。
      
  en: >
      The multithreaded loopback server and its configuration file reader.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.605Z"
fingerprint: 6c2e840eb4584e5b87955eb2ea24baaf4194c875a3f3cb897255b978282779b3
source:
  - path: "backend/process.py"
    line: 508
    end_line: 534
apis:
  - protocol: file
    path: "backend/process.py#BackendHttpServer"
    description:
      zh: >
          在回环地址上以多线程提供 HTTP 服务的服务器类。
          
      en: >
          The server class that serves HTTP over multiple threads on a loopback address.
          
---
