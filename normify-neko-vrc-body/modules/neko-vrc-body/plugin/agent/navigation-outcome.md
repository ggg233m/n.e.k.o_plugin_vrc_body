---
uid: 9e3a0217
id: neko-vrc-body.plugin.agent.navigation-outcome
parent: neko-vrc-body.plugin.agent
name: {zh: "导航结果回报", en: "Navigation Outcome"}
description:
  zh: >
      把导航目标的结果推回宿主对话，让 agent 知道这次是否奏效。
      
  en: >
      Pushing the outcome of a navigation goal back into the host conversation so the agent learns whether it worked.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.849Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 933
    end_line: 1063
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._push_navigation_outcome"
    description:
      zh: >
          把导航结果推回宿主对话。
          
      en: >
          Push a navigation outcome back into the host conversation.
          
---
