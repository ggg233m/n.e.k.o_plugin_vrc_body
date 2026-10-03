---
uid: 9e3d020e
id: neko-vrc-body.backend.nav-loop.closer.state
parent: neko-vrc-body.backend.nav-loop.closer
name: {zh: "闭合器状态与位姿存取", en: "Closer State"}
description:
  zh: >
      节点与位姿存取、漂移半径，以及按回环位移修正单个或全部位姿。
      
  en: >
      Node and pose storage, drift radius, and correction of a single pose or all poses by the loop displacement.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.624Z"
fingerprint: 0d562774afcbca2772a8abb5b0d193fdfb369fdf0074750d5fb80d71774492c7
source:
  - path: "backend/nav_loop.py"
    line: 211
    end_line: 313
apis:
  - protocol: file
    path: "backend/nav_loop.py#LoopCloser.pose"
    description:
      zh: >
          按节点编号读取单个位姿。
          
      en: >
          Reads the pose of a single node by index.
          
  - protocol: file
    path: "backend/nav_loop.py#LoopCloser.correct"
    description:
      zh: >
          按回环位移修正全部位姿。
          
      en: >
          Corrects all poses by the loop closure displacement.
          
  - protocol: file
    path: "backend/nav_loop.py#LoopCloser._drift_radius"
    description:
      zh: >
          估计累计漂移的半径。
          
      en: >
          Estimates the radius of accumulated drift.
          
---
