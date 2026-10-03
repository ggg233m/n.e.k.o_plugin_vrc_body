---
uid: "9e420026"
id: neko-vrc-body.backend.nav-mapping.mapper.ray-veto
parent: neko-vrc-body.backend.nav-mapping.mapper
name: {zh: "行走轨迹射线否决", en: "Walked-trail ray veto"}
description:
  zh: >
      记录身体实际走过的轨迹，并据此否决射线清除出的假障碍。
      
  en: >
      Records the trail the body actually walked and vetoes obstacles falsely produced by ray clearing along that trail.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.633Z"
fingerprint: caf6dd30bbd13ba420ba62eb770627a77f528a9314d72c2456ff0d2b6210709f
source:
  - path: "backend/nav_mapping.py"
    line: 660
    end_line: 742
apis:
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper.add_trail"
    description:
      zh: >
          记录身体实际走过的轨迹。
          
      en: >
          Records the trail the body actually walked.
          
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper.walked"
    description:
      zh: >
          回答某个格子是否被走过。
          
      en: >
          Answers whether a cell was walked through.
          
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper._ray_veto"
    description:
      zh: >
          否决射线清除产生的假障碍。
          
      en: >
          Vetoes obstacles falsely produced by ray clearing.
          
---
