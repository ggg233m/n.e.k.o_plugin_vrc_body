---
uid: 9e3a0311
id: neko-vrc-body.body-kernel.motion.rotations
parent: neko-vrc-body.body-kernel.motion
name: {zh: "腕部与掌心旋转", en: "Wrist and Palm Rotations"}
description:
  zh: >
      AnyaDance 体内坐标下的掌心、腕部与欧拉旋转构造，以及跟随手臂方向的变体。
      
  en: >
      Palm, wrist and Euler rotation construction in the AnyaDance body axes, plus the directed variant.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.788Z"
fingerprint: 87b9c47e51e1edf49546d81a698b1088f6f5798e561623d5a8cbfbb63a6e69be
source:
  - path: "motion.py"
    line: 139
    end_line: 199
apis:
  - protocol: file
    path: "motion.py#euler_rotation"
    description:
      zh: >
          按 AnyaDance 坐标轴构造的体内 yaw/pitch/roll 旋转。
          
      en: >
          Body-local yaw/pitch/roll rotation using the AnyaDance axes.
          
  - protocol: file
    path: "motion.py#palm_rotation"
    description:
      zh: >
          某个手部姿态名对应的掌心朝向。
          
      en: >
          Palm orientation for a hand pose name.
          
  - protocol: file
    path: "motion.py#wrist_rotation"
    description:
      zh: >
          某个手部姿态名对应的腕部旋转。
          
      en: >
          Wrist rotation for a hand pose name.
          
  - protocol: file
    path: "motion.py#directed_wrist_rotation"
    description:
      zh: >
          先跟随手臂方向、再叠加掌心与腕部局部偏移。
          
      en: >
          Wrist rotation that first follows the arm direction.
          
---
