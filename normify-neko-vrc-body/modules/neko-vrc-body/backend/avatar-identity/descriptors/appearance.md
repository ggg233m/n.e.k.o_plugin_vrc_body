---
uid: 9e41000c
id: neko-vrc-body.backend.avatar-identity.descriptors.appearance
parent: neko-vrc-body.backend.avatar-identity.descriptors
name: {zh: "外观描述子", en: "Appearance Descriptor"}
description:
  zh: >
      轻量、亮度相对稳定的外观描述子，由颜色比例直方图与空间颜色统计组成。
      
  en: >
      A lightweight appearance descriptor that stays relatively stable under brightness change, built from a color-ratio histogram and spatial color statistics.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.575Z"
fingerprint: 89b5d2d1b91411d8de7f700e453b25567d25e099f70a2b5e51581f661c2a2793
source:
  - path: "backend/avatar_identity.py"
    line: 36
    end_line: 146
apis:
  - protocol: file
    path: "backend/avatar_identity.py#appearance_descriptor"
    description:
      zh: >
          从检测裁剪图计算亮度相对稳定的外观描述子。
          
      en: >
          Computes a brightness-stable appearance descriptor from a detection crop.
          
  - protocol: file
    path: "backend/avatar_identity.py#_normalized_bbox"
    description:
      zh: >
          把边界框归一化到单位区间以便跨尺度比较。
          
      en: >
          Normalizes a bounding box into the unit interval for cross-scale comparison.
          
---
