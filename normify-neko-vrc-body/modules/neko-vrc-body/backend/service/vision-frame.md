---
uid: 9e3e0015
id: neko-vrc-body.backend.service.vision-frame
parent: neko-vrc-body.backend.service
name: {zh: "视觉帧服务", en: "Vision Frame Serving"}
description:
  zh: >
      把最新帧送给模型，带显式的新鲜度上限与可选叠加。
      
  en: >
      Serving the latest frame to the model with an explicit freshness bound and optional overlay.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.758Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2379
    end_line: 2403
apis:
  - protocol: file
    path: "backend/service.py#BackendService.vision_frame"
    description:
      zh: >
          返回最新帧，可带检测叠加与最大帧龄。
          
      en: >
          Return the latest frame, optionally with detection overlay and a max age.
          
---
