---
uid: 9e3a0042
id: neko-vrc-body.backend.pose-math
parent: neko-vrc-body.backend
name: {zh: "位姿数学", en: "Pose Maths"}
description:
  zh: >
      航位推算的共享纯数学：带符号偏航换算、本地位移旋入世界系与位置累加。
      
  en: >
      The shared pure maths for dead reckoning: signed yaw conversion, local-to-world rotation and position advance.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.725Z"
fingerprint: bdaf4b31a765338ef6f0a8ef27c69c01050f6954194e239e7daff1b4acd7d8b1
source:
  - path: "backend/pose_math.py"
    line: 67
    end_line: 107
apis:
  - protocol: file
    path: "backend/pose_math.py#advance"
    description:
      zh: >
          在线与离线共用的唯一一份航位推算数学实现。
          
      en: >
          The single implementation of dead-reckoning maths shared by online and offline paths.
          
  - protocol: file
    path: "backend/pose_math.py#rotate_local_to_world"
    description:
      zh: >
          把化身本地位移矢量旋进世界系。
          
      en: >
          Rotate an avatar-local displacement into the world frame.
          
  - protocol: file
    path: "backend/pose_math.py#yaw_radians"
    description:
      zh: >
          在显式偏航符号约定下把角度转成带符号弧度。
          
      en: >
          Convert degrees to signed radians under an explicit yaw sign convention.
          
---
