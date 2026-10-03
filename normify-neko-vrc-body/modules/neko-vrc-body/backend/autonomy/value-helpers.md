---
uid: 9e3f0007
id: neko-vrc-body.backend.autonomy.value-helpers
parent: neko-vrc-body.backend.autonomy
name: {zh: "目标取值归一化助手", en: "Goal Value Normalizers"}
description:
  zh: >
      允许的目标种类、选择器键、约束键与方向分数的归一化；坏键不应让整段操作失败。
      
  en: >
      Normalization of allowed goal kinds, selector keys, constraint keys, and direction scores, where a bad key must not fail the whole operation.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.574Z"
fingerprint: 956f89ee80cd18326626bcfed51b616a94c813b4ec5dd9fa58499c2a52baf96a
source:
  - path: "backend/autonomy.py"
    line: 17
    end_line: 169
apis:
  - protocol: file
    path: "backend/autonomy.py#_normalize_selector"
    description:
      zh: >
          把任意选择器映射归一化为受支持的键集合。
          
      en: >
          Normalizes an arbitrary selector mapping into the supported set of keys.
          
  - protocol: file
    path: "backend/autonomy.py#_normalize_constraints"
    description:
      zh: >
          丢弃非法键后归一化约束字典。
          
      en: >
          Normalizes a constraint mapping after dropping invalid keys.
          
  - protocol: file
    path: "backend/autonomy.py#_normalize_direction_scores"
    description:
      zh: >
          把方向分数归一化到有界区间并过滤异常项。
          
      en: >
          Normalizes direction scores into a bounded range and filters out outliers.
          
---
