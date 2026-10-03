---
uid: "9e420029"
id: neko-vrc-body.backend.nav-mapping.frontier
parent: neko-vrc-body.backend.nav-mapping
name: {zh: "探索边界目标", en: "Exploration frontiers"}
description:
  zh: >
      探索目标：紧挨未知区边界、位于可走中心区且与起点连通。
      
  en: >
      Exploration goals that hug the unknown-region border, lie in the walkable center area, and are connected to the start.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.627Z"
fingerprint: caf6dd30bbd13ba420ba62eb770627a77f528a9314d72c2456ff0d2b6210709f
source:
  - path: "backend/nav_mapping.py"
    line: 905
    end_line: 982
apis:
  - protocol: file
    path: "backend/nav_mapping.py#frontiers"
    description:
      zh: >
          返回当前栅格的探索边界目标。
          
      en: >
          Returns the exploration frontiers of the current grid.
          
  - protocol: file
    path: "backend/nav_mapping.py#_geodesic"
    description:
      zh: >
          计算用于给边界目标排序的测地距离场。
          
      en: >
          Computes the geodesic distance field used to rank frontiers.
          
---
