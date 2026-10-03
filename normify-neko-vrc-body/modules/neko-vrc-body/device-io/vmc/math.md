---
uid: 9e3a0351
id: neko-vrc-body.device-io.vmc.math
parent: neko-vrc-body.device-io.vmc
name: {zh: "变换数学", en: "Transform Maths"}
description:
  zh: >
      向量与四元数数学，以及人形正向运动学所依赖的变换复合。
      
  en: >
      Vector and quaternion maths plus the transform composition the humanoid forward kinematics depends on.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.832Z"
fingerprint: a49ba23fb5ec0e9ebaa1ef04e8b0a3ded0b99c3a792d7b1568bce88cb271c578
source:
  - path: "vmc_idle.py"
    line: 112
    end_line: 181
apis:
  - protocol: file
    path: "vmc_idle.py#_compose"
    description:
      zh: >
          把局部骨骼变换复合到父节点上。
          
      en: >
          Compose a local bone transform onto its parent.
          
  - protocol: file
    path: "vmc_idle.py#_vmc_transform"
    description:
      zh: >
          把一条 VMC OSC 参数元组解码成具名骨骼变换。
          
      en: >
          Decode one VMC OSC argument tuple into named bone transforms.
          
  - protocol: file
    path: "vmc_idle.py#_quat_angle"
    description:
      zh: >
          两个四元数之间的最短夹角。
          
      en: >
          Shortest angular distance between two quaternions.
          
---
