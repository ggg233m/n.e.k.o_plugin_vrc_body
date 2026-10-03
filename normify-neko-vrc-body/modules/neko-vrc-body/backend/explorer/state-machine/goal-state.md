---
uid: 9e50001d
id: neko-vrc-body.backend.explorer.state-machine.goal-state
parent: neko-vrc-body.backend.explorer.state-machine
name: {zh: "探索目标状态", en: "Explore Goal State"}
description:
  zh: >
      目标签名、重置与实体匹配规则。
      
  en: >
      Goal signature, reset and the entity-matching rules.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.599Z"
fingerprint: 3ceb238589b7465b0375c659e6db1f5f984144bfe544a81d29d28423b1db0ad9
source:
  - path: "backend/explorer.py"
    line: 65
    end_line: 165
apis:
  - protocol: file
    path: "backend/explorer.py#ExplorerStateMachine.reset"
    description:
      zh: >
          重置探索目标状态。
          
      en: >
          Resets the explore goal state.
          
  - protocol: file
    path: "backend/explorer.py#ExplorerStateMachine._goal_signature"
    description:
      zh: >
          构造当前目标的签名。
          
      en: >
          Builds the signature of the current goal.
          
  - protocol: file
    path: "backend/explorer.py#ExplorerStateMachine._matches"
    description:
      zh: >
          判断某个实体是否匹配目标。
          
      en: >
          Tells whether an entity matches the goal.
          
---
