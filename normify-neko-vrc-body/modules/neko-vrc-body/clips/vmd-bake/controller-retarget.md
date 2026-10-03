---
uid: 9e3a0392
id: neko-vrc-body.clips.vmd-bake.controller-retarget
parent: neko-vrc-body.clips.vmd-bake
name: {zh: "控制器重定向", en: "Controller Retargeting"}
description:
  zh: >
      从手臂链导出控制器旋转，含让手腕朝向正确的扭转标定。
      
  en: >
      Deriving controller rotation from the arm chain, including the twist calibration that makes a wrist face the right way.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.810Z"
fingerprint: ff195c90f5686de6f36a074e773db33f01c0cb0477451740a419014ee45f511b
source:
  - path: "vmd_bake.py"
    line: 205
    end_line: 231
apis:
  - protocol: file
    path: "vmd_bake.py#_controller_reference_twist"
    description:
      zh: >
          沿手臂链计算控制器参考扭转方向。
          
      en: >
          Reference twist direction for a controller along an arm chain.
          
  - protocol: file
    path: "vmd_bake.py#_controller_rotation"
    description:
      zh: >
          由扭转与基底导出控制器旋转。
          
      en: >
          Derive a controller rotation from a twist and basis.
          
---
