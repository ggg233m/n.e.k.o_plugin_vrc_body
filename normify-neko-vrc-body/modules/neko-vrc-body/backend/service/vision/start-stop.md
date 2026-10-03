---
uid: 9e3e0009
id: neko-vrc-body.backend.service.vision.start-stop
parent: neko-vrc-body.backend.service.vision
name: {zh: "视觉启停", en: "Vision Start and Stop"}
description:
  zh: >
      启停感知，含世界状态存储必须知道的状态迁移。
      
  en: >
      Starting and stopping perception, including the state transitions the world store needs to hear about.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.761Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 1975
    end_line: 2093
apis:
  - protocol: file
    path: "backend/service.py#BackendService.vision_start"
    description:
      zh: >
          启动感知。
          
      en: >
          Start perception.
          
  - protocol: file
    path: "backend/service.py#BackendService.vision_stop"
    description:
      zh: >
          停止感知。
          
      en: >
          Stop perception.
          
---
