---
uid: 9e3a0210
id: neko-vrc-body.plugin.agent.tool-registration
parent: neko-vrc-body.plugin.agent
name: {zh: "Agent 条目注册", en: "Agent Entry Registration"}
description:
  zh: >
      向宿主 agent 注册表发布与撤回工具定义及行为指令。
      
  en: >
      Publishing and withdrawing the tool definitions and behaviour instructions with the host agent registry.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.854Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 510
    end_line: 658
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._register_agent_entries"
    description:
      zh: >
          向宿主 agent 发布 24 个工具定义与指令。
          
      en: >
          Publish the 24 tool definitions and instructions to the host agent.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._unregister_agent_entries"
    description:
      zh: >
          撤回全部已发布的 agent 条目。
          
      en: >
          Withdraw every published agent entry.
          
---
