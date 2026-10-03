---
uid: "9e500005"
id: neko-vrc-body.backend.navigator.config.value-helpers
parent: neko-vrc-body.backend.navigator.config
name: {zh: "决策取值助手", en: "Decision Value Helpers"}
description:
  zh: >
      决策前的取值助手：有限性、收敛与映射。
      
  en: >
      Value helpers used before a decision: finiteness, clamping and mapping.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.668Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 285
    end_line: 300
apis:
  - protocol: file
    path: "backend/navigator.py#_finite"
    description:
      zh: >
          判断取值是否有限。
          
      en: >
          Tells whether a value is finite.
          
  - protocol: file
    path: "backend/navigator.py#_clamp"
    description:
      zh: >
          把取值收敛到给定区间。
          
      en: >
          Clamps a value into the given range.
          
---
