---
uid: 9e3d020c
id: neko-vrc-body.backend.nav-loop.relative-pose
parent: neko-vrc-body.backend.nav-loop
name: {zh: "相对位姿求解", en: "Relative Pose"}
description:
  zh: >
      由特征求出相对位姿，以及位姿图的稀疏拉普拉斯求解。
      
  en: >
      Derives the relative pose from features and solves the pose graph sparsely via a Laplacian system.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.626Z"
fingerprint: 0d562774afcbca2772a8abb5b0d193fdfb369fdf0074750d5fb80d71774492c7
source:
  - path: "backend/nav_loop.py"
    line: 126
    end_line: 190
apis:
  - protocol: file
    path: "backend/nav_loop.py#relative_pose"
    description:
      zh: >
          由特征对应求出两关键帧间的相对位姿。
          
      en: >
          Solves the relative pose between two keyframes from feature correspondences.
          
  - protocol: file
    path: "backend/nav_loop.py#_solve_laplacian"
    description:
      zh: >
          稀疏求解位姿图的拉普拉斯方程。
          
      en: >
          Solves the pose graph Laplacian equation with a sparse solver.
          
---
