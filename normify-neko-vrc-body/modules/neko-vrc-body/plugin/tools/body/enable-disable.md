---
uid: 9e3a0224
id: neko-vrc-body.plugin.tools.body.enable-disable
parent: neko-vrc-body.plugin.tools.body
name: {zh: "启用与关闭", en: "Enable and Disable"}
description:
  zh: >
      打开与关闭身体输出这两个最简开关。
      
  en: >
      The two trivial switches that turn the body output on and off.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.885Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2244
    end_line: 2251
apis:
  - protocol: rpc
    path: "body_enable"
    description:
      zh: >
          工具：启用身体输出。
          
      en: >
          Tool: enable the body output.
          
  - protocol: rpc
    path: "body_disable"
    description:
      zh: >
          工具：关闭身体输出。
          
      en: >
          Tool: disable the body output.
          
---
