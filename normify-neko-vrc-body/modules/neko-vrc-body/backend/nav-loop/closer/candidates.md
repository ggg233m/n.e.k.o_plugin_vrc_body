---
uid: 9e3d020f
id: neko-vrc-body.backend.nav-loop.closer.candidates
parent: neko-vrc-body.backend.nav-loop.closer
name: {zh: "回环候选筛选", en: "Closer Candidates"}
description:
  zh: >
      加入关键帧，并经词袋候选与几何验证筛出回环。
      
  en: >
      Adds keyframes and filters loop closures through bag-of-words candidates and geometric verification.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.622Z"
fingerprint: 0d562774afcbca2772a8abb5b0d193fdfb369fdf0074750d5fb80d71774492c7
source:
  - path: "backend/nav_loop.py"
    line: 315
    end_line: 443
apis:
  - protocol: file
    path: "backend/nav_loop.py#LoopCloser.add_keyframe"
    description:
      zh: >
          把一个关键帧加入回环检测库。
          
      en: >
          Adds a keyframe into the loop closure database.
          
  - protocol: file
    path: "backend/nav_loop.py#LoopCloser._candidates"
    description:
      zh: >
          取出回环候选关键帧对。
          
      en: >
          Retrieves candidate keyframe pairs for loop closure.
          
  - protocol: file
    path: "backend/nav_loop.py#LoopCloser._verify"
    description:
      zh: >
          用几何一致性验证回环候选。
          
      en: >
          Verifies loop closure candidates by geometric consistency.
          
---
