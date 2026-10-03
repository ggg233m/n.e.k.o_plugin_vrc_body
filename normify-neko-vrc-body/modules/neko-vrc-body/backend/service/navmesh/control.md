---
uid: 9e3e0005
id: neko-vrc-body.backend.service.navmesh.control
parent: neko-vrc-body.backend.service.navmesh
name: {zh: "Navmesh 控制", en: "Navmesh Control"}
description:
  zh: >
      在线 navmesh 导航器的控制面：状态、覆盖、启停、goto、探索与取消。
      
  en: >
      Control surface for the online navmesh navigator: status, coverage, start, stop, goto, explore and cancel.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.743Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 545
    end_line: 577
apis:
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_status"
    description:
      zh: >
          报告在线 navmesh 状态，可选附带栅格化结果。
          
      en: >
          Report the online navmesh state, optionally with the rasterised grid.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_coverage"
    description:
      zh: >
          报告累积覆盖视图。
          
      en: >
          Report the accumulated coverage view.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_start"
    description:
      zh: >
          启动在线导航器。
          
      en: >
          Start the online navigator.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_goto"
    description:
      zh: >
          导航到某个世界坐标。
          
      en: >
          Navigate to a world coordinate.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_explore"
    description:
      zh: >
          启动一次边界探索。
          
      en: >
          Start a frontier exploration.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_cancel"
    description:
      zh: >
          取消当前导航目标。
          
      en: >
          Cancel the current navigation goal.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_stop"
    description:
      zh: >
          停止在线导航器。
          
      en: >
          Stop the online navigator.
          
---
