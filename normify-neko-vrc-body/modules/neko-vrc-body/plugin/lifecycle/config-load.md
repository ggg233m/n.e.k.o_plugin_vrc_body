---
uid: 9e3a0202
id: neko-vrc-body.plugin.lifecycle.config-load
parent: neko-vrc-body.plugin.lifecycle
name: {zh: "配置加载", en: "Config Load"}
description:
  zh: >
      异步加载宿主设置并构造出经过校验的 PluginConfig。
      
  en: >
      Asynchronously load the host settings and build the validated PluginConfig.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.874Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 381
    end_line: 391
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._load_config"
    description:
      zh: >
          加载并校验插件设置，产出 PluginConfig。
          
      en: >
          Load and validate the plugin settings into a PluginConfig.
          
---
