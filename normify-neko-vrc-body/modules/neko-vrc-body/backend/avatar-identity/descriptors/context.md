---
uid: 9e41000d
id: neko-vrc-body.backend.avatar-identity.descriptors.context
parent: neko-vrc-body.backend.avatar-identity.descriptors
name: {zh: "背景上下文指纹", en: "Background Context Fingerprint"}
description:
  zh: >
      低分辨率背景指纹，只在外观歧义时判断摄像机是否仍在同一视角。
      
  en: >
      A low-resolution background fingerprint used only when appearance is ambiguous, to judge whether the camera is still in the same view.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.575Z"
fingerprint: 89b5d2d1b91411d8de7f700e453b25567d25e099f70a2b5e51581f661c2a2793
source:
  - path: "backend/avatar_identity.py"
    line: 149
    end_line: 247
apis:
  - protocol: file
    path: "backend/avatar_identity.py#_context_descriptor"
    description:
      zh: >
          把帧压成低分辨率背景指纹以供视角比对。
          
      en: >
          Compresses a frame into a low-resolution background fingerprint for view comparison.
          
  - protocol: file
    path: "backend/avatar_identity.py#_bbox_distance"
    description:
      zh: >
          计算两个边界框之间的几何距离用于上下文消歧。
          
      en: >
          Computes the geometric distance between two boxes for context disambiguation.
          
---
