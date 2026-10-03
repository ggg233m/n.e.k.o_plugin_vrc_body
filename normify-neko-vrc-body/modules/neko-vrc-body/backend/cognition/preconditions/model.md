---
uid: 9e3f0002
id: neko-vrc-body.backend.cognition.preconditions.model
parent: neko-vrc-body.backend.cognition.preconditions
name: {zh: "前置条件模型", en: "Precondition Model"}
description:
  zh: >
      动作对最新世界状态的显式、可序列化约束。
      
  en: >
      Explicit, serializable constraints that an action places on the latest world state.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.589Z"
fingerprint: ef89d358bc9669d29d55dc2f3e13d67ab8ffb2d1c9f55ccdf7de4175419d1d8a
source:
  - path: "backend/cognition.py"
    line: 112
    end_line: 217
apis:
  - protocol: file
    path: "backend/cognition.py#WorldPrecondition"
    description:
      zh: >
          动作前置条件的数据结构定义。
          
      en: >
          The data structure describing one action precondition.
          
  - protocol: file
    path: "backend/cognition.py#WorldPrecondition.from_mapping"
    description:
      zh: >
          从外部映射构造前置条件，忽略未知或非法字段。
          
      en: >
          Builds a precondition from an external mapping, ignoring unknown or invalid fields.
          
  - protocol: file
    path: "backend/cognition.py#WorldPrecondition.to_dict"
    description:
      zh: >
          把前置条件序列化为纯 JSON 结构。
          
      en: >
          Serializes the precondition into a plain JSON structure.
          
---
