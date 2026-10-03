---
uid: 9e3d0203
id: neko-vrc-body.backend.nav-grid.grid.build
parent: neko-vrc-body.backend.nav-grid.grid
name: {zh: "可行走中心区构建", en: "Walkable Center Region Build"}
description:
  zh: >
      把三态栅格分类为可行走中心区，并把目标吸附到中心区。
      
  en: >
      Classifies a tri-state grid into a walkable center region and snaps targets into it.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.619Z"
fingerprint: ab228e994b036051b250d76a3fe27c96a2aff1be0f388d31a08b24394a6bc5a1
source:
  - path: "backend/nav_grid.py"
    line: 110
    end_line: 173
apis:
  - protocol: file
    path: "backend/nav_grid.py#NavGrid.build"
    description:
      zh: >
          由三态栅格构建可行走中心区。
          
      en: >
          Builds the walkable center region from a tri-state grid.
          
  - protocol: file
    path: "backend/nav_grid.py#NavGrid.snap_to_center"
    description:
      zh: >
          把目标点吸附到可行走中心区。
          
      en: >
          Snaps a target point into the walkable center region.
          
---
