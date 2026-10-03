---
uid: 9e3e0008
id: neko-vrc-body.backend.service.vision.attach
parent: neko-vrc-body.backend.service.vision
name: {zh: "视觉接入", en: "Vision Attachment"}
description:
  zh: >
      注入点：宿主项目可在此提供自己的画面源与检测器，替代配置中的实现。
      
  en: >
      The injection point where a host project supplies its own frame source and detector instead of the configured ones.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.759Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 1916
    end_line: 1973
apis:
  - protocol: file
    path: "backend/service.py#BackendService.attach_vision"
    description:
      zh: >
          注入外部提供的画面源与检测器。
          
      en: >
          Inject an externally supplied frame source and detector.
          
---
