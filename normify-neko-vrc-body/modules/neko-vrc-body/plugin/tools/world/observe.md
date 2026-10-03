---
uid: 9e3a0230
id: neko-vrc-body.plugin.tools.world.observe
parent: neko-vrc-body.plugin.tools.world
name: {zh: "世界观测", en: "World Observe"}
description:
  zh: >
      读取当前世界快照：实体、事件、不确定性与导航状态，可按需裁剪。
      
  en: >
      Reading the current world snapshot: entities, events, uncertainties and navigation state, optionally narrowed.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.909Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2739
    end_line: 2800
apis:
  - protocol: rpc
    path: "world_observe"
    description:
      zh: >
          工具：读取当前世界快照、实体与不确定性。
          
      en: >
          Tool: read the current world snapshot, entities and uncertainties.
          
---
