---
uid: "9e500021"
id: neko-vrc-body.backend.nav-mapping.mapper.accumulator.ray-encode
parent: neko-vrc-body.backend.nav-mapping.mapper.accumulator
name: {zh: "射线体素编码", en: "Ray Voxel Encoding"}
description:
  zh: >
      射线体素的编码与同步。
      
  en: >
      Encoding and synchronisation of ray voxels.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.629Z"
fingerprint: caf6dd30bbd13ba420ba62eb770627a77f528a9314d72c2456ff0d2b6210709f
source:
  - path: "backend/nav_mapping.py"
    line: 564
    end_line: 658
apis:
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper._encode_ray"
    description:
      zh: >
          把射线编码成体素格。
          
      en: >
          Encodes a ray into voxel cells.
          
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper._sync_ray"
    description:
      zh: >
          让射线票数与当前位姿同步。
          
      en: >
          Synchronises ray votes with the current poses.
          
---
