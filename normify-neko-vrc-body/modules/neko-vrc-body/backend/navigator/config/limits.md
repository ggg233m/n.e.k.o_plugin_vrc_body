---
uid: "9e500002"
id: neko-vrc-body.backend.navigator.config.limits
parent: neko-vrc-body.backend.navigator.config
name: {zh: "循环安全上限", en: "Loop Safety Limits"}
description:
  zh: >
      十赫兹局部循环的安全上限常量与 NavigatorConfig。
      
  en: >
      Safety-limit constants and NavigatorConfig, the bounds the 10 Hz local loop must respect.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.666Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 28
    end_line: 235
apis:
  - protocol: file
    path: "backend/navigator.py#NavigatorConfig"
    description:
      zh: >
          十赫兹循环必须遵守的有界上限。
          
      en: >
          The bounded limits that the 10 Hz loop must respect.
          
---
