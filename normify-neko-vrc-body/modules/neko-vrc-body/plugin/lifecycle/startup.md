---
uid: 9e3a0203
id: neko-vrc-body.plugin.lifecycle.startup
parent: neko-vrc-body.plugin.lifecycle
name: {zh: "启动", en: "Start-up"}
description:
  zh: >
      on_startup：加载配置、启动后端客户端与驱动中继，并注册 agent 条目。
      
  en: >
      on_startup: load config, start the backend client, drivers and relays, and register agent entries.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.877Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 394
    end_line: 488
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin.on_startup"
    description:
      zh: >
          拉起后端客户端、驱动、中继、世界上下文桥与 agent 注册项。
          
      en: >
          Bring up the backend client, drivers, relays, the world-context bridge and the agent entries.
          
deps:
  - kind: call
    to: neko-vrc-body.backend.client.connection
    label: {zh: "启动后端子进程", en: "Start the backend subprocess"}
  - kind: call
    to: neko-vrc-body.config.plugin-config
    label: {zh: "加载并校验设置", en: "Load and validate the settings"}
  - kind: call
    to: neko-vrc-body.plugin.agent.tool-registration
    label: {zh: "注册工具定义", en: "Register the tool definitions"}
  - kind: call
    to: neko-vrc-body.plugin.agent.world-context-bridge
    label: {zh: "启动世界上下文桥", en: "Start the world-context bridge"}
---
