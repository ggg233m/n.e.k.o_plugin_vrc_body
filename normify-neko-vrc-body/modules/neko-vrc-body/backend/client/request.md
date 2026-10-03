---
uid: 9e3c010a
id: neko-vrc-body.backend.client.request
parent: neko-vrc-body.backend.client
name: {zh: "请求通道", en: "Request Channels"}
description:
  zh: >
      请求通道：持久快速通道、一次性请求，以及「优先使用调用方提供的控制通道」这条规则。
      
  en: >
      The request channels: a persistent fast path, one-shot requests, and the rule for preferring a supplied control channel.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.586Z"
fingerprint: 8f6d5a4ca8fb37bed50fe66446dfcb2eb54d67a9caca0125f881ac24d9844a7e
source:
  - path: "backend/client.py"
    line: 240
    end_line: 394
apis:
  - protocol: file
    path: "backend/client.py#BackendClient.fast_request"
    description:
      zh: >
          读取自某修订以来的世界增量。
          
      en: >
          Read the world delta since a revision.
          
  - protocol: file
    path: "backend/client.py#BackendClient.request"
    description:
      zh: >
          向后端发出一次请求。
          
      en: >
          Issue one request to the backend.
          
  - protocol: file
    path: "backend/client.py#_control_request"
    description:
      zh: >
          调用方提供持久通道时优先使用它。
          
      en: >
          Prefer the persistent control channel when the caller supplies one.
          
deps:
  - kind: call
    to: neko-vrc-body.backend.ipc.http-handler.post-routes
    label: {zh: "经回环 POST", en: "POST over loopback"}
---
