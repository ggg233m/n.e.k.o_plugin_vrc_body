---
uid: "9e420027"
id: neko-vrc-body.backend.nav-mapping.mapper.poses
parent: neko-vrc-body.backend.nav-mapping.mapper
name: {zh: "位姿更新与锚定", en: "Pose update and anchoring"}
description:
  zh: >
      接受回环修正后的位姿，并把世界坐标锚定到最近关键帧。
      
  en: >
      Accepts loop-closure corrected poses and anchors world coordinates to the nearest keyframe.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.631Z"
fingerprint: caf6dd30bbd13ba420ba62eb770627a77f528a9314d72c2456ff0d2b6210709f
source:
  - path: "backend/nav_mapping.py"
    line: 744
    end_line: 778
apis:
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper.update_poses"
    description:
      zh: >
          接受回环修正后的位姿。
          
      en: >
          Accepts the loop-closure corrected poses.
          
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper.anchor"
    description:
      zh: >
          把世界坐标锚定到最近关键帧。
          
      en: >
          Anchors a world coordinate to the nearest keyframe.
          
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper.resolve"
    description:
      zh: >
          基于锚定关键帧解析一次查询。
          
      en: >
          Resolves a query against the anchored keyframe.
          
---
