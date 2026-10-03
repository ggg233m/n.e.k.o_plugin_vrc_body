---
uid: 9e3d0207
id: neko-vrc-body.backend.nav-follow.follower
parent: neko-vrc-body.backend.nav-follow
name: {zh: "路径跟随器", en: "Path Follower"}
description:
  zh: >
      Pure pursuit 路径跟随，输出前进轴与期望偏航角速度。
      
  en: >
      Pure pursuit path following, producing a forward axis value and a desired yaw rate.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.618Z"
fingerprint: e04443cf978aa8fa70bfa2a90b855b2665df6c03489b8102b037e97a24315555
source:
  - path: "backend/nav_follow.py"
    line: 136
    end_line: 226
apis:
  - protocol: file
    path: "backend/nav_follow.py#PathFollower.step"
    description:
      zh: >
          推进一步跟随状态并输出驱动指令。
          
      en: >
          Advances the follower by one step and returns the drive commands.
          
  - protocol: file
    path: "backend/nav_follow.py#PathFollower._point_at"
    description:
      zh: >
          按前视距离取出路径上的目标点。
          
      en: >
          Returns the path point at a given look-ahead distance.
          
---
