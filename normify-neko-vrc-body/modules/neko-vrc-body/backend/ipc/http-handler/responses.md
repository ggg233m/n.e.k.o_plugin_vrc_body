---
uid: 9e3c0103
id: neko-vrc-body.backend.ipc.http-handler.responses
parent: neko-vrc-body.backend.ipc.http-handler
name: {zh: "响应写出辅助", en: "Response Writer Helpers"}
description:
  zh: >
      响应写出辅助：JSON、二进制、MJPEG 流、静态资源分发与请求体读取。
      
  en: >
      Response writing helpers: JSON, binary, MJPEG streaming, static asset serving and request body reading.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.604Z"
fingerprint: 6c2e840eb4584e5b87955eb2ea24baaf4194c875a3f3cb897255b978282779b3
source:
  - path: "backend/process.py"
    line: 139
    end_line: 216
apis:
  - protocol: file
    path: "backend/process.py#BackendRequestHandler._json"
    description:
      zh: >
          写出 JSON 响应。
          
      en: >
          Writes a JSON response.
          
  - protocol: file
    path: "backend/process.py#BackendRequestHandler._serve_ui"
    description:
      zh: >
          按路由名分发 UI 静态资源。
          
      en: >
          Serves a UI static asset selected by route name.
          
---
