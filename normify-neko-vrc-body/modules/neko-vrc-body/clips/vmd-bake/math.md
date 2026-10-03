---
uid: 9e3a0390
id: neko-vrc-body.clips.vmd-bake.math
parent: neko-vrc-body.clips.vmd-bake
name: {zh: "向量与四元数数学", en: "Vector and Quaternion Maths"}
description:
  zh: >
      重定向路径的向量与四元数数学：加减、缩放、点积、叉积、归一化与旋转。
      
  en: >
      Vector and quaternion maths for the retargeting path: add, subtract, scale, dot, cross, normalise and rotate.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.811Z"
fingerprint: ff195c90f5686de6f36a074e773db33f01c0cb0477451740a419014ee45f511b
source:
  - path: "vmd_bake.py"
    line: 70
    end_line: 133
apis:
  - protocol: file
    path: "vmd_bake.py#_quat_multiply"
    description:
      zh: >
          重定向路径共用的向量与四元数数学。
          
      en: >
          Vector and quaternion maths shared by the retargeting path.
          
  - protocol: file
    path: "vmd_bake.py#_quat_rotate"
    description:
      zh: >
          用四元数旋转向量。
          
      en: >
          Rotate a vector by a quaternion.
          
  - protocol: file
    path: "vmd_bake.py#_normalize_vec"
    description:
      zh: >
          归一化向量，退化时使用回退值。
          
      en: >
          Normalise a vector, falling back when degenerate.
          
---
