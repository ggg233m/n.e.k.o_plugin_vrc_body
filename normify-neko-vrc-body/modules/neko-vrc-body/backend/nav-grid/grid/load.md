---
uid: 9e3d0202
id: neko-vrc-body.backend.nav-grid.grid.load
parent: neko-vrc-body.backend.nav-grid.grid
name: {zh: "栅格载入与坐标换算", en: "Grid Load and Coordinate Conversion"}
description:
  zh: >
      从 PGM 三态栅格载入，并做世界坐标与格坐标的双向换算。
      
  en: >
      Loads a PGM tri-state grid and converts between world and cell coordinates both ways.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.620Z"
fingerprint: ab228e994b036051b250d76a3fe27c96a2aff1be0f388d31a08b24394a6bc5a1
source:
  - path: "backend/nav_grid.py"
    line: 59
    end_line: 107
apis:
  - protocol: file
    path: "backend/nav_grid.py#NavGrid.load"
    description:
      zh: >
          从 PGM 三态栅格文件载入网格数据。
          
      en: >
          Loads grid data from a PGM tri-state grid file.
          
  - protocol: file
    path: "backend/nav_grid.py#NavGrid.to_cell"
    description:
      zh: >
          世界坐标换算为格坐标。
          
      en: >
          Converts world coordinates into cell coordinates.
          
  - protocol: file
    path: "backend/nav_grid.py#NavGrid.to_world"
    description:
      zh: >
          格坐标换算为世界坐标。
          
      en: >
          Converts cell coordinates into world coordinates.
          
---
