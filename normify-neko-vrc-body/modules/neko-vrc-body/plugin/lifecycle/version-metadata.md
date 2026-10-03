---
uid: 9e3a0200
id: neko-vrc-body.plugin.lifecycle.version-metadata
parent: neko-vrc-body.plugin.lifecycle
name: {zh: "版本与常量", en: "Version and Constants"}
description:
  zh: >
      从 plugin.toml 发现版本号，以及模块级的时限与体积常量。
      
  en: >
      Version discovery from plugin.toml and the module-level timing and size constants.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.879Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 48
    end_line: 64
  - path: "__init__.py"
    line: 181
    end_line: 198
apis:
  - protocol: file
    path: "__init__.py#_plugin_version"
    description:
      zh: >
          从 plugin.toml 读取权威版本号，缺失时回落到已知可用字面量。
          
      en: >
          Read the authoritative version from plugin.toml, falling back to a known-good literal.
          
  - protocol: file
    path: "__init__.py#_MAIN_SERVER_PORT"
    description:
      zh: >
          插件自带主服务绑定的回环端口。
          
      en: >
          The loopback port the plugin's own main server binds.
          
---
