---
uid: 9e3a0100
id: neko-vrc-body.llm-contract.tool-defs.preconditions
parent: neko-vrc-body.llm-contract.tool-defs
name: {zh: "世界前置条件", en: "World Preconditions"}
description:
  zh: >
      可复用的世界前置条件，被那些必须对照最新世界状态求值的工具共享。
      
  en: >
      Reusable world preconditions shared by tools that must be evaluated against the latest world state.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.839Z"
fingerprint: 348077cad7b4ebac68be08404f65a6094ebab38baa302e5e51c5b85edc242009
source:
  - path: "tool_defs.py"
    line: 3
    end_line: 28
apis:
  - protocol: file
    path: "tool_defs.py#WORLD_PRECONDITIONS"
    description:
      zh: >
          可复用的世界前置条件，挂在依赖「化身当前能看到什么」的工具上。
          
      en: >
          Reusable world preconditions attached to tools that depend on what the avatar can currently see.
          
---
