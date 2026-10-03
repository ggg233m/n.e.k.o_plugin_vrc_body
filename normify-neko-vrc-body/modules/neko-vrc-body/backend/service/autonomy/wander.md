---
uid: 9e3e0010
id: neko-vrc-body.backend.service.autonomy.wander
parent: neko-vrc-body.backend.service.autonomy
name: {zh: "闲逛步进", en: "Wander Step"}
description:
  zh: >
      一步自由漫游：结合模型偏好与记忆选定方位，迈出一步，再记录结果。
      
  en: >
      One free-roam step: pick a bearing from the model's preferences and the memory, step, then record the outcome.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.731Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 1297
    end_line: 1419
apis:
  - protocol: file
    path: "backend/service.py#BackendService.autonomy_wander_step"
    description:
      zh: >
          推进一步有界的自由漫游。
          
      en: >
          Advance one bounded free-roam step.
          
---
