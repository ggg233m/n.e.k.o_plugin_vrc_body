---
uid: "9e420024"
id: neko-vrc-body.backend.nav-mapping.mapper.init
parent: neko-vrc-body.backend.nav-mapping.mapper
name: {zh: "映射器初始化与关键帧", en: "Mapper init and keyframes"}
description:
  zh: >
      映射器构造、加入与丢弃关键帧，以及全局相机离地高度的更新。
      
  en: >
      Mapper construction, adding and dropping keyframes, and the update of the global camera height above ground.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.631Z"
fingerprint: caf6dd30bbd13ba420ba62eb770627a77f528a9314d72c2456ff0d2b6210709f
source:
  - path: "backend/nav_mapping.py"
    line: 293
    end_line: 425
apis:
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper.add_keyframe"
    description:
      zh: >
          把一个带点的关键帧加入地图。
          
      en: >
          Adds one keyframe with its points to the map.
          
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper.drop_points"
    description:
      zh: >
          丢弃某个关键帧已存的点。
          
      en: >
          Drops the stored points of one keyframe.
          
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper._update_cam_h"
    description:
      zh: >
          更新全局相机离地高度。
          
      en: >
          Updates the global camera height above ground.
          
---
