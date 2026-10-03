---
uid: 9e3d0200
id: neko-vrc-body.backend.nav-grid.types
parent: neko-vrc-body.backend.nav-grid
name: {zh: "导航栅格类型", en: "Navigation Grid Types"}
description:
  zh: >
      栅格元数据（原点、分辨率、每格米数）与路径合约。
      
  en: >
      Grid metadata (origin, resolution, meters per cell) and the path contract.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.622Z"
fingerprint: ab228e994b036051b250d76a3fe27c96a2aff1be0f388d31a08b24394a6bc5a1
source:
  - path: "backend/nav_grid.py"
    line: 32
    end_line: 56
apis:
  - protocol: file
    path: "backend/nav_grid.py#GridMeta"
    description:
      zh: >
          栅格元数据：原点、分辨率与每格米数。
          
      en: >
          Grid metadata: origin, resolution, and meters per cell.
          
  - protocol: file
    path: "backend/nav_grid.py#PathContract"
    description:
      zh: >
          路径合约：规划结果的约定与约束。
          
      en: >
          Path contract: the agreed shape and constraints of a planned path.
          
---
