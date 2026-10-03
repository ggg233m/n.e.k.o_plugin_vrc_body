---
uid: 9e3a0103
id: neko-vrc-body.llm-contract.tool-defs.body-io
parent: neko-vrc-body.llm-contract.tool-defs
name: {zh: "身体控制与状态工具", en: "Body Control and Status Tools"}
description:
  zh: >
      原始 VRChat 输入、紧急停止、中立复位与身体状态读取的 Schema。
      
  en: >
      Schemas for raw VRChat input, emergency stop, neutral reset and the body status read-out.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.837Z"
fingerprint: 348077cad7b4ebac68be08404f65a6094ebab38baa302e5e51c5b85edc242009
source:
  - path: "tool_defs.py"
    line: 147
    end_line: 423
apis:
  - protocol: file
    path: "tool_defs.py#BODY_VRCHAT_INPUT"
    description:
      zh: >
          工具 Schema：通过 AnyaDance 桥直接驱动 VRChat 化身输入。
          
      en: >
          Tool schema: drive raw VRChat avatar input through the AnyaDance bridge.
          
  - protocol: file
    path: "tool_defs.py#BODY_STOP"
    description:
      zh: >
          工具 Schema：停止所有身体动作。
          
      en: >
          Tool schema: stop all body motion.
          
  - protocol: file
    path: "tool_defs.py#BODY_RESET"
    description:
      zh: >
          工具 Schema：把身体复位到中立 T-pose。
          
      en: >
          Tool schema: reset the body to the neutral T-pose.
          
  - protocol: file
    path: "tool_defs.py#BODY_STATUS"
    description:
      zh: >
          工具 Schema：读取身体运行时快照。
          
      en: >
          Tool schema: read the body runtime snapshot.
          
---
