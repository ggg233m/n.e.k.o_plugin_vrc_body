---
uid: 9e3a0211
id: neko-vrc-body.plugin.agent.tool-execution
parent: neko-vrc-body.plugin.agent
name: {zh: "工具结果规范化", en: "Tool Result Normalisation"}
description:
  zh: >
      工具调用的结果规范化，以及三个轻量的世界观测与导航 agent 工具。
      
  en: >
      Result normalisation for tool calls, plus the three thin world-observation and navigation agent tools.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.852Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 661
    end_line: 698
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._execution_result"
    description:
      zh: >
          把工具执行结果规范化给宿主，可要求必须已完成。
          
      en: >
          Normalise a tool execution result for the host, optionally requiring completion.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._agent_observe_vrchat_world"
    description:
      zh: >
          Agent 工具：观测当前 VRChat 世界。
          
      en: >
          Agent tool: observe the current VRChat world.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._agent_scan_vrchat_surroundings"
    description:
      zh: >
          Agent 工具：扫描化身周围环境。
          
      en: >
          Agent tool: scan the avatar's surroundings.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._agent_navigate_vrchat_world"
    description:
      zh: >
          Agent 工具：在 VRChat 世界中导航。
          
      en: >
          Agent tool: navigate the VRChat world.
          
---
