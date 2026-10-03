---
uid: 9e3d0204
id: neko-vrc-body.backend.nav-grid.grid.plan
parent: neko-vrc-body.backend.nav-grid.grid
name: {zh: "路径规划与拉直", en: "Path Planning and String Pulling"}
description:
  zh: >
      A* 规划、代价地图与拉直；输出带合约的路径。
      
  en: >
      A* planning, cost map, and string pulling; emits a path carrying the contract.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.621Z"
fingerprint: ab228e994b036051b250d76a3fe27c96a2aff1be0f388d31a08b24394a6bc5a1
source:
  - path: "backend/nav_grid.py"
    line: 175
    end_line: 265
apis:
  - protocol: file
    path: "backend/nav_grid.py#NavGrid.plan"
    description:
      zh: >
          规划一条从起点到终点的路径。
          
      en: >
          Plans a path from a start point to a goal.
          
  - protocol: file
    path: "backend/nav_grid.py#NavGrid._astar"
    description:
      zh: >
          在代价地图上执行 A* 搜索。
          
      en: >
          Runs A* search over the cost map.
          
  - protocol: file
    path: "backend/nav_grid.py#NavGrid._string_pull"
    description:
      zh: >
          对路径做拉直平滑处理。
          
      en: >
          Smooths a path by string pulling.
          
---
