---
uid: 9e3a0101
id: neko-vrc-body.llm-contract.tool-defs.body-posture
parent: neko-vrc-body.llm-contract.tool-defs
name: {zh: "身体姿态工具", en: "Body Posture Tools"}
description:
  zh: >
      启用/关闭身体以及预备命名全身姿态的 Schema。
      
  en: >
      Schemas for enabling/disabling the body and arming a named whole-body pose.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.838Z"
fingerprint: 348077cad7b4ebac68be08404f65a6094ebab38baa302e5e51c5b85edc242009
source:
  - path: "tool_defs.py"
    line: 29
    end_line: 71
apis:
  - protocol: file
    path: "tool_defs.py#BODY_ENABLE"
    description:
      zh: >
          工具 Schema：启用身体输出。
          
      en: >
          Tool schema: enable the body output.
          
  - protocol: file
    path: "tool_defs.py#BODY_DISABLE"
    description:
      zh: >
          工具 Schema：关闭身体输出。
          
      en: >
          Tool schema: disable the body output.
          
  - protocol: file
    path: "tool_defs.py#BODY_ARM_POSE"
    description:
      zh: >
          工具 Schema：预备一个命名的全身姿态。
          
      en: >
          Tool schema: arm a named whole-body pose.
          
---
