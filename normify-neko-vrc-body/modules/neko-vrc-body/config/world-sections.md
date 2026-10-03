---
uid: 9e3a0023
id: neko-vrc-body.config.world-sections
parent: neko-vrc-body.config
name: {zh: "自主与世界配置段", en: "Autonomy and World Sections"}
description:
  zh: >
      会话授权默认值、可持久化世界事实的策略与手动世界身份约定。
      
  en: >
      Session authorization defaults, the persistable-world-fact policy and the manual world-identity contract.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.816Z"
fingerprint: 55757bfaa420ee4e8c4479614d38026395bc79552cfe74aec0d249ef070ee23f
source:
  - path: "config.py"
    line: 199
    end_line: 229
apis:
  - protocol: file
    path: "config.py#AutonomyConfig"
    description:
      zh: >
          会话授权默认值；自主操作绝不隐式启用。
          
      en: >
          Session authorization defaults; autonomy is never implicitly enabled.
          
  - protocol: file
    path: "config.py#WorldMemoryConfig"
    description:
      zh: >
          允许持久化哪些可复用的世界事实，以及写到哪里。
          
      en: >
          Which reusable world facts may be persisted, and where.
          
  - protocol: file
    path: "config.py#WorldModelConfig"
    description:
      zh: >
          手动世界身份；world_key 绝不自动探测，也不持久化。
          
      en: >
          Manual world identity; the world key is never auto-detected or persisted.
          
---
