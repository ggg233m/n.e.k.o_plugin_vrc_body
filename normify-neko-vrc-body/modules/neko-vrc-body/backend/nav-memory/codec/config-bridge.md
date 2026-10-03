---
uid: 9e70000c
id: neko-vrc-body.backend.nav-memory.codec.config-bridge
parent: neko-vrc-body.backend.nav-memory.codec
name: {zh: "配置桥接", en: "Config Bridge"}
description:
  zh: >
      逐字段从插件的 NavmeshMemoryConfig 构造 MemoryConfig。
      
  en: >
      Building a MemoryConfig from the plugin's NavmeshMemoryConfig, field by field.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.636Z"
fingerprint: 8ac3ac2937f0825e7d07c1267ffa2e59000b81e0743552690708cb1b776bbb0e
source:
  - path: "backend/nav_memory.py"
    line: 515
    end_line: 517
apis:
  - protocol: file
    path: "backend/nav_memory.py#config_from_plugin"
    description:
      zh: >
          从插件配置构造记忆库配置。
          
      en: >
          Build a memory config from the plugin config.
          
---
