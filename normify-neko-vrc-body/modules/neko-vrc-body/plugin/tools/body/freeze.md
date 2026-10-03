---
uid: 9e3a022c
id: neko-vrc-body.plugin.tools.body.freeze
parent: neko-vrc-body.plugin.tools.body
name: {zh: "紧急冻结", en: "Emergency Freeze"}
description:
  zh: >
      紧急冻结路径，以及决定模型能否解除冻结的两个工具。
      
  en: >
      The emergency freeze path and the two tools that decide whether the model may lift it.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.887Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2565
    end_line: 2618
apis:
  - protocol: rpc
    path: "body_freeze"
    description:
      zh: >
          工具：由面板触发的紧急冻结。
          
      en: >
          Tool: emergency freeze, applied by the panel.
          
  - protocol: rpc
    path: "llm_freeze_allow"
    description:
      zh: >
          工具：模型有把握时解除冻结。
          
      en: >
          Tool: lift the freeze when the model is confident.
          
  - protocol: rpc
    path: "llm_freeze_deny"
    description:
      zh: >
          工具：维持冻结。
          
      en: >
          Tool: keep the freeze in place.
          
---
