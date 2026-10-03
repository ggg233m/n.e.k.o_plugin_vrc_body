---
uid: 9e3d0210
id: neko-vrc-body.backend.nav-loop.closer.optimize
parent: neko-vrc-body.backend.nav-loop.closer
name: {zh: "位姿图优化", en: "Closer Optimization"}
description:
  zh: >
      迭代最小化平移位姿图残差，并回报回环统计。
      
  en: >
      Iteratively minimizes translation pose graph residuals and reports loop closure statistics.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.623Z"
fingerprint: 0d562774afcbca2772a8abb5b0d193fdfb369fdf0074750d5fb80d71774492c7
source:
  - path: "backend/nav_loop.py"
    line: 444
    end_line: 516
apis:
  - protocol: file
    path: "backend/nav_loop.py#LoopCloser.optimize"
    description:
      zh: >
          迭代最小化平移位姿图残差。
          
      en: >
          Iteratively minimizes translation pose graph residuals.
          
  - protocol: file
    path: "backend/nav_loop.py#LoopCloser.status"
    description:
      zh: >
          回报回环检测与优化的统计信息。
          
      en: >
          Reports loop closure detection and optimization statistics.
          
---
