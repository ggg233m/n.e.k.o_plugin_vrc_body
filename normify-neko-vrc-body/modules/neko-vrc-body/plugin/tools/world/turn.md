---
uid: 9e3a0232
id: neko-vrc-body.plugin.tools.world.turn
parent: neko-vrc-body.plugin.tools.world
name: {zh: "化身转向", en: "Turn Avatar"}
description:
  zh: >
      按有界的相对偏航角转动化身，并受当前朝向整定状态门控。
      
  en: >
      Turning the avatar by a bounded relative yaw, gated against the current heading settle state.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.910Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 3011
    end_line: 3073
apis:
  - protocol: rpc
    path: "body_turn"
    description:
      zh: >
          工具：按有界的相对偏航角转动化身。
          
      en: >
          Tool: turn the avatar by a bounded relative yaw.
          
---
