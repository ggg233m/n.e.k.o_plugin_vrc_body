---
uid: 9e3a0020
id: neko-vrc-body.config.helpers
parent: neko-vrc-body.config
name: {zh: "取值助手", en: "Coercion Helpers"}
description:
  zh: >
      配置段读取器与有限浮点、比例、整数、布尔强制转换助手，含旧 min_box_ratio 回退。
      
  en: >
      Section readers and the finite/ratio/integer/boolean coercion helpers, including the legacy min_box_ratio fallback.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.812Z"
fingerprint: 55757bfaa420ee4e8c4479614d38026395bc79552cfe74aec0d249ef070ee23f
source:
  - path: "config.py"
    line: 11
    end_line: 82
apis:
  - protocol: file
    path: "config.py#_finite_float"
    description:
      zh: >
          把取值强制转换成范围内的有限浮点。
          
      en: >
          Coerce a value to a finite float inside an explicit range.
          
  - protocol: file
    path: "config.py#_axis_box_ratio"
    description:
      zh: >
          解析单轴最小框比例，并兼容旧版共用的 min_box_ratio 键。
          
      en: >
          Resolve a per-axis minimum box ratio, honouring the legacy shared min_box_ratio key.
          
  - protocol: file
    path: "config.py#_bounded_int"
    description:
      zh: >
          把取值强制转换成落在显式范围内的整数。
          
      en: >
          Coerce a value to an integer clamped to an explicit range.
          
  - protocol: file
    path: "config.py#_boolean"
    description:
      zh: >
          把取值强制转换成布尔，容忍常见的真值写法。
          
      en: >
          Coerce a value to a boolean, accepting the usual truthy spellings.
          
---
