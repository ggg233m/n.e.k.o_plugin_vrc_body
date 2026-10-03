---
uid: 9e3a005a
id: neko-vrc-body.backend.nav-loop
parent: neko-vrc-body.backend
name: {zh: "回环闭合", en: "Loop Closure"}
description:
  zh: >
      回环检测加只优化平移的位姿图，让航位推算在回到旧地时重新对齐。
      
  en: >
      Loop closure plus a translation-only pose graph, so dead reckoning re-aligns whenever the avatar revisits a place.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.625Z"
fingerprint: 0d562774afcbca2772a8abb5b0d193fdfb369fdf0074750d5fb80d71774492c7
source:
  - path: "backend/nav_loop.py"
    line: 36
    end_line: 516
---
