---
uid: 9e3a0391
id: neko-vrc-body.clips.vmd-bake.basis
parent: neko-vrc-body.clips.vmd-bake
name: {zh: "基底求解", en: "Basis Solving"}
description:
  zh: >
      重定向的基底求解：挑选稳定的垂直方向、构造正交标架，并完成局部到世界的映射。
      
  en: >
      Basis solving for the retarget: picking a stable perpendicular, building an orthonormal frame and mapping local to world.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.809Z"
fingerprint: ff195c90f5686de6f36a074e773db33f01c0cb0477451740a419014ee45f511b
source:
  - path: "vmd_bake.py"
    line: 129
    end_line: 202
apis:
  - protocol: file
    path: "vmd_bake.py#_basis"
    description:
      zh: >
          从两个参考方向构造正交基。
          
      en: >
          Build an orthonormal basis from two reference directions.
          
  - protocol: file
    path: "vmd_bake.py#_matrix_to_quat"
    description:
      zh: >
          把旋转矩阵转成四元数。
          
      en: >
          Convert a rotation matrix into a quaternion.
          
  - protocol: file
    path: "vmd_bake.py#_basis_mapping"
    description:
      zh: >
          把局部基底映射到世界基底。
          
      en: >
          Map a local basis onto a world basis.
          
---
