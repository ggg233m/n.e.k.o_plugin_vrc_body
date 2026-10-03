---
uid: "9e420021"
id: neko-vrc-body.backend.nav-mapping.voxel.ray
parent: neko-vrc-body.backend.nav-mapping.voxel
name: {zh: "射线体素", en: "Ray voxels"}
description:
  zh: >
      一个关键帧的障碍高度体素，以及累加器布局所需的行去重。
      
  en: >
      The obstacle height voxels of one keyframe and the row de-duplication required by the accumulator layout.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.636Z"
fingerprint: caf6dd30bbd13ba420ba62eb770627a77f528a9314d72c2456ff0d2b6210709f
source:
  - path: "backend/nav_mapping.py"
    line: 181
    end_line: 220
apis:
  - protocol: file
    path: "backend/nav_mapping.py#_ray_voxels"
    description:
      zh: >
          枚举一条障碍射线穿过的体素。
          
      en: >
          Enumerates the voxels along an obstacle ray.
          
  - protocol: file
    path: "backend/nav_mapping.py#unique_rows"
    description:
      zh: >
          对累加器布局所需的行做去重。
          
      en: >
          De-duplicates the rows required by the accumulator layout.
          
---
