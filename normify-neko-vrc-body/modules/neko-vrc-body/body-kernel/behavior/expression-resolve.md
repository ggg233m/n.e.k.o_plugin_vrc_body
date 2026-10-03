---
uid: 9e3a0307
id: neko-vrc-body.body-kernel.behavior.expression-resolve
parent: neko-vrc-body.body-kernel.behavior
name: {zh: "表情解析", en: "Expression Resolution"}
description:
  zh: >
      把语义意图解析成表情档位，以及防止叠加层削弱显式工具的准入守卫。
      
  en: >
      Resolving a semantic intent into an expression profile, and the admission guard that keeps overlays from weakening explicit tools.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.781Z"
fingerprint: 1a12074fd870f4fbc2e33a1a6015fbc4a415817d225662e7f7d1613b78c7a42a
source:
  - path: "behavior.py"
    line: 81
    end_line: 117
apis:
  - protocol: file
    path: "behavior.py#resolve_expression"
    description:
      zh: >
          把自由文本意图解析成具体的表情档位。
          
      en: >
          Resolve a free-text intent into a concrete expression profile.
          
  - protocol: file
    path: "behavior.py#expression_admission"
    description:
      zh: >
          判定一个低优先级表情叠加层此刻是否可以启动。
          
      en: >
          Decide whether a low-priority expression overlay may start right now.
          
---
