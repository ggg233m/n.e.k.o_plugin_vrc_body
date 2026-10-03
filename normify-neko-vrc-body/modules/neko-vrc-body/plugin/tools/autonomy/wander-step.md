---
uid: 9e3a0237
id: neko-vrc-body.plugin.tools.autonomy.wander-step
parent: neko-vrc-body.plugin.tools.autonomy
name: {zh: "闲逛步进", en: "Wander Step"}
description:
  zh: >
      一步有界的自由漫游：选方向、短暂移动，再结合方向记忆重新评估。
      
  en: >
      One bounded free-roam step: pick a direction, move briefly, then reassess against the direction memory.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.882Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 3233
    end_line: 3272
apis:
  - protocol: rpc
    path: "vrc_wander_step"
    description:
      zh: >
          工具：在当前授权下推进一步有界的自由漫游。
          
      en: >
          Tool: advance one bounded free-roam step under the current authorization.
          
---
