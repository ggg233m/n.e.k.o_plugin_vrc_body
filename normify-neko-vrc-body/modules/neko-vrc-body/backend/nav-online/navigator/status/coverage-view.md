---
uid: 9e50001b
id: neko-vrc-body.backend.nav-online.navigator.status.coverage-view
parent: neko-vrc-body.backend.nav-online.navigator.status
name: {zh: "覆盖率视图", en: "Coverage View"}
description:
  zh: >
      覆盖率的累积与世界坐标到覆盖格的换算。
      
  en: >
      Coverage accumulation and the conversion from world coordinates to coverage cells.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.651Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 1151
    end_line: 1287
apis:
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator.coverage_view"
    description:
      zh: >
          返回对外的覆盖率视图。
          
      en: >
          Returns the outward coverage view.
          
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator._update_coverage"
    description:
      zh: >
          累积覆盖率计数。
          
      en: >
          Accumulates the coverage counts.
          
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator._world_to_cov_cell"
    description:
      zh: >
          把世界坐标换算成覆盖格。
          
      en: >
          Converts world coordinates into a coverage cell.
          
---
