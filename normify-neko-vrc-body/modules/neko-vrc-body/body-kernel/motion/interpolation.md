---
uid: 9e3a0310
id: neko-vrc-body.body-kernel.motion.interpolation
parent: neko-vrc-body.body-kernel.motion
name: {zh: "插值", en: "Interpolation"}
description:
  zh: >
      收敛、缓动、向量与四元数插值，以及整帧插值。
      
  en: >
      Clamping, easing, vector and quaternion interpolation, and whole-frame interpolation.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.787Z"
fingerprint: 87b9c47e51e1edf49546d81a698b1088f6f5798e561623d5a8cbfbb63a6e69be
source:
  - path: "motion.py"
    line: 39
    end_line: 136
apis:
  - protocol: file
    path: "motion.py#clamp01"
    description:
      zh: >
          把取值收敛到 0..1。
          
      en: >
          Clamp a value into 0..1.
          
  - protocol: file
    path: "motion.py#smoothstep"
    description:
      zh: >
          平滑步进缓动。
          
      en: >
          Smoothstep easing.
          
  - protocol: file
    path: "motion.py#quat_slerp"
    description:
      zh: >
          两个四元数之间的球面线性插值。
          
      en: >
          Spherical linear interpolation between two quaternions.
          
  - protocol: file
    path: "motion.py#interpolate_frame"
    description:
      zh: >
          在两帧完整身体姿态之间插值。
          
      en: >
          Interpolate two whole body frames.
          
  - protocol: file
    path: "motion.py#quat_between_vectors"
    description:
      zh: >
          把一个矢量旋到另一个矢量上的最短旋转。
          
      en: >
          The shortest rotation mapping one vector onto another.
          
---
