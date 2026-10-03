---
uid: 9e3c0113
id: neko-vrc-body.backend.client.remote-vision
parent: neko-vrc-body.backend.client
name: {zh: "视觉代理", en: "Vision Proxy"}
description:
  zh: >
      视觉代理：世界增量、取帧、语义请求与提交、外部观测注入。
      
  en: >
      The vision proxy: world deltas, frame fetching, semantic request and commit, and external observation ingest.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.585Z"
fingerprint: 8f6d5a4ca8fb37bed50fe66446dfcb2eb54d67a9caca0125f881ac24d9844a7e
source:
  - path: "backend/client.py"
    line: 797
    end_line: 977
apis:
  - protocol: file
    path: "backend/client.py#RemoteVision.delta"
    description:
      zh: >
          从远端视觉代理读取世界增量。
          
      en: >
          Reads world deltas from the remote vision agent.
          
  - protocol: file
    path: "backend/client.py#RemoteVision.frame"
    description:
      zh: >
          从远端视觉代理取回最新画面。
          
      en: >
          Fetches the latest frame from the remote vision agent.
          
  - protocol: file
    path: "backend/client.py#RemoteVision.semantic_request"
    description:
      zh: >
          轮询远端视觉代理上的语义请求。
          
      en: >
          Polls a semantic request on the remote vision agent.
          
  - protocol: file
    path: "backend/client.py#RemoteVision.semantic_commit"
    description:
      zh: >
          向远端视觉代理提交语义结果。
          
      en: >
          Commits a semantic result on the remote vision agent.
          
  - protocol: file
    path: "backend/client.py#RemoteVision.ingest"
    description:
      zh: >
          向远端视觉代理注入外部观测。
          
      en: >
          Injects external observations into the remote vision agent.
          
---
