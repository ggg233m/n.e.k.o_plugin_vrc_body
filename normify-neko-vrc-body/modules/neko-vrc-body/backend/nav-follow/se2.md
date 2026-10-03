---
uid: 9e3d0205
id: neko-vrc-body.backend.nav-follow.se2
parent: neko-vrc-body.backend.nav-follow
name: {zh: "SE(2) 位姿运算", en: "SE(2) Pose Math"}
description:
  zh: >
      SE(2) 位姿的复合、求逆与角度折回。
      
  en: >
      SE(2) pose composition, inversion, and angle wrapping.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.619Z"
fingerprint: e04443cf978aa8fa70bfa2a90b855b2665df6c03489b8102b037e97a24315555
source:
  - path: "backend/nav_follow.py"
    line: 23
    end_line: 35
apis:
  - protocol: file
    path: "backend/nav_follow.py#_compose"
    description:
      zh: >
          复合两个 SE(2) 位姿。
          
      en: >
          Composes two SE(2) poses.
          
  - protocol: file
    path: "backend/nav_follow.py#_inverse"
    description:
      zh: >
          求 SE(2) 位姿的逆。
          
      en: >
          Inverts an SE(2) pose.
          
  - protocol: file
    path: "backend/nav_follow.py#wrap"
    description:
      zh: >
          把角度折回 (-pi, pi] 区间。
          
      en: >
          Wraps an angle back into the range (-pi, pi].
          
---
