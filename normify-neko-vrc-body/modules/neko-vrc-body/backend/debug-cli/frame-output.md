---
uid: 9e3e0024
id: neko-vrc-body.backend.debug-cli.frame-output
parent: neko-vrc-body.backend.debug-cli
name: {zh: "请求与帧输出", en: "Request and Frame Output"}
description:
  zh: >
      发出请求，并把 base64 帧落盘以保持终端可读。
      
  en: >
      Issuing a request and writing a base64 frame to disk so the terminal stays readable.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.591Z"
fingerprint: ea1de34602a67e7be6b4c263a343c3d70a69e31f189f484adadc872b211d42a8
source:
  - path: "backend/debug_cli.py"
    line: 113
    end_line: 166
apis:
  - protocol: file
    path: "backend/debug_cli.py#_write_frame"
    description:
      zh: >
          把 base64 帧落盘，并从打印结果中摘掉它。
          
      en: >
          Write a base64 frame to disk and strip it from the printed result.
          
  - protocol: file
    path: "backend/debug_cli.py#request"
    description:
      zh: >
          向在线后端发出一次请求。
          
      en: >
          Issue one request to the live backend.
          
---
