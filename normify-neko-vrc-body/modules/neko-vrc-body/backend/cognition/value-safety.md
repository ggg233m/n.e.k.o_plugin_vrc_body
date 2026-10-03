---
uid: 9e3f0000
id: neko-vrc-body.backend.cognition.value-safety
parent: neko-vrc-body.backend.cognition
name: {zh: "取值安全助手", en: "Value Safety Helpers"}
description:
  zh: >
      保证适配器与 LLM 之间传递的数据有界且 JSON 安全的取值助手。
      
  en: >
      Bounded, JSON-safe value helpers that keep the data passed between adapters and the LLM within limits.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.590Z"
fingerprint: ef89d358bc9669d29d55dc2f3e13d67ab8ffb2d1c9f55ccdf7de4175419d1d8a
source:
  - path: "backend/cognition.py"
    line: 20
    end_line: 108
apis:
  - protocol: file
    path: "backend/cognition.py#_safe"
    description:
      zh: >
          把任意输入收敛为有界且 JSON 安全的表示。
          
      en: >
          Coerces any input into a bounded, JSON-safe representation.
          
  - protocol: file
    path: "backend/cognition.py#_confidence"
    description:
      zh: >
          归一化并裁剪置信度到 0..1 区间。
          
      en: >
          Normalizes and clamps a confidence value into the 0..1 range.
          
  - protocol: file
    path: "backend/cognition.py#_aliased_string"
    description:
      zh: >
          按别名表读取字符串字段，缺失时回退默认值。
          
      en: >
          Reads a string field through an alias table, falling back to a default when it is absent.
          
---
