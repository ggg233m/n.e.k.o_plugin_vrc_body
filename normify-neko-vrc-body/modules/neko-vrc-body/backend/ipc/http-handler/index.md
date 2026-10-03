---
uid: 9e3c0101
id: neko-vrc-body.backend.ipc.http-handler
parent: neko-vrc-body.backend.ipc
name: {zh: "HTTP 请求处理器", en: "HTTP Request Handler"}
description:
  zh: >
      基于 BaseHTTPRequestHandler 的回环请求处理器，拆分为鉴权、响应写出、只读 GET 路由与命令 POST 路由。
      
  en: >
      The loopback request handler built on BaseHTTPRequestHandler, split into authorization, response writing, read-only GET routes and command POST routes.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.603Z"
fingerprint: 6c2e840eb4584e5b87955eb2ea24baaf4194c875a3f3cb897255b978282779b3
source:
  - path: "backend/process.py"
    line: 122
    end_line: 505
---
