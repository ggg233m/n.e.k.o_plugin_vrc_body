---
uid: 9e3a0201
id: neko-vrc-body.plugin.lifecycle.tool-arg-validation
parent: neko-vrc-body.plugin.lifecycle
name: {zh: "工具参数校验", en: "Tool Argument Validation"}
description:
  zh: >
      对来自宿主模型的参数执行的收敛与枚举校验。
      
  en: >
      Clamping and enum validation applied to arguments arriving from the host model.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.878Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 67
    end_line: 105
apis:
  - protocol: file
    path: "__init__.py#_enum"
    description:
      zh: >
          按允许集合校验字符串参数。
          
      en: >
          Validate a string argument against an allowed set.
          
  - protocol: file
    path: "__init__.py#_number"
    description:
      zh: >
          把数值参数收敛到显式范围内。
          
      en: >
          Clamp a numeric argument to an explicit range.
          
  - protocol: file
    path: "__init__.py#_integer"
    description:
      zh: >
          把整数参数收敛到显式范围内。
          
      en: >
          Clamp an integer argument to an explicit range.
          
  - protocol: file
    path: "__init__.py#_boolean"
    description:
      zh: >
          强制转换布尔工具参数。
          
      en: >
          Coerce a boolean tool argument.
          
---
