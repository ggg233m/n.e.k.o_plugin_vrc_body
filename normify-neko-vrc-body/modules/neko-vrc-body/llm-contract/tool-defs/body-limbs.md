---
uid: 9e3a0102
id: neko-vrc-body.llm-contract.tool-defs.body-limbs
parent: neko-vrc-body.llm-contract.tool-defs
name: {zh: "肢体与表情工具", en: "Limb and Expression Tools"}
description:
  zh: >
      手部姿态、伸手抓取、命名手势与语义表情的 Schema。
      
  en: >
      Schemas for hand poses, reach-and-grab, named gestures and semantic expression.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.837Z"
fingerprint: 348077cad7b4ebac68be08404f65a6094ebab38baa302e5e51c5b85edc242009
source:
  - path: "tool_defs.py"
    line: 72
    end_line: 146
apis:
  - protocol: file
    path: "tool_defs.py#BODY_HAND"
    description:
      zh: >
          工具 Schema：设定手部姿态。
          
      en: >
          Tool schema: set a hand pose.
          
  - protocol: file
    path: "tool_defs.py#BODY_REACH_AND_GRAB"
    description:
      zh: >
          工具 Schema：伸手够向目标并抓取。
          
      en: >
          Tool schema: reach towards a target and grab.
          
  - protocol: file
    path: "tool_defs.py#BODY_GESTURE"
    description:
      zh: >
          工具 Schema：播放一个命名手势。
          
      en: >
          Tool schema: play a named gesture.
          
  - protocol: file
    path: "tool_defs.py#BODY_EXPRESS"
    description:
      zh: >
          工具 Schema：表达一种语义情绪。
          
      en: >
          Tool schema: express a semantic emotion.
          
---
