---
uid: 9e3a021c
id: neko-vrc-body.plugin.agent.tool-reassert
parent: neko-vrc-body.plugin.agent
name: {zh: "工具重述看门狗", en: "Tool Re-assert Watchdog"}
description:
  zh: >
      看门狗：按固定间隔重新发布被宿主注册表静默丢弃的工具定义。
      
  en: >
      Watchdog that re-asserts tool definitions the host registry silently dropped, on a fixed interval.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.853Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 1495
    end_line: 1627
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._reassert_llm_tools"
    description:
      zh: >
          重新发布宿主注册表丢失的工具，返回恢复数量。
          
      en: >
          Re-publish any tool the host registry has dropped, returning how many were restored.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._llm_tool_watch_once"
    description:
      zh: >
          对宿主工具注册表执行一次看门狗检查。
          
      en: >
          One watchdog pass over the host tool registry.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._host_tool_registry_names"
    description:
      zh: >
          询问宿主当前暴露了哪些工具名。
          
      en: >
          Ask the host which tool names it currently exposes.
          
---
