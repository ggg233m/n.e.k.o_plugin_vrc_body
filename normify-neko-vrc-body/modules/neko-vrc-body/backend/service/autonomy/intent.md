---
uid: 9e3e0012
id: neko-vrc-body.backend.service.autonomy.intent
parent: neko-vrc-body.backend.service.autonomy
name: {zh: "意图执行", en: "Intent Execution"}
description:
  zh: >
      把高层意图翻译成导航器的工作：候选枚举、目标下发与停止。
      
  en: >
      Translating a high-level intent into navigator work: candidate enumeration, goal dispatch and stopping.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.729Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 1529
    end_line: 1914
apis:
  - protocol: file
    path: "backend/service.py#BackendService.autonomy_intent"
    description:
      zh: >
          对着导航器执行一条高层意图。
          
      en: >
          Execute a high-level intent against the navigator.
          
  - protocol: file
    path: "backend/service.py#BackendService._semantic_navigation_candidates"
    description:
      zh: >
          枚举某个选择器可能匹配的导航候选。
          
      en: >
          Enumerate the navigation candidates a selector could match.
          
  - protocol: file
    path: "backend/service.py#BackendService.autonomy_stop"
    description:
      zh: >
          停止自主目标。
          
      en: >
          Stop the autonomy goal.
          
---
