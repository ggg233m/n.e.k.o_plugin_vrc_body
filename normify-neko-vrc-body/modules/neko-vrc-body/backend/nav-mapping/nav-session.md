---
uid: 9e42002a
id: neko-vrc-body.backend.nav-mapping.nav-session
parent: neko-vrc-body.backend.nav-mapping
name: {zh: "增量地图导航会话", en: "Incremental map navigation session"}
description:
  zh: >
      增量地图上的去某处或探索会话：每次地图更新都重新栅格化、重规划并换一个跟随器。
      
  en: >
      A goto or explore session on the incremental map that re-rasterizes, re-plans, and swaps the follower on every map update.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.633Z"
fingerprint: caf6dd30bbd13ba420ba62eb770627a77f528a9314d72c2456ff0d2b6210709f
source:
  - path: "backend/nav_mapping.py"
    line: 985
    end_line: 1149
apis:
  - protocol: file
    path: "backend/nav_mapping.py#NavSession.explore"
    description:
      zh: >
          在增量地图上启动一次探索会话。
          
      en: >
          Starts an explore session on the incremental map.
          
  - protocol: file
    path: "backend/nav_mapping.py#NavSession.compute"
    description:
      zh: >
          计算本会话当前的路径。
          
      en: >
          Computes the current path for the session.
          
  - protocol: file
    path: "backend/nav_mapping.py#NavSession.goto"
    description:
      zh: >
          在增量地图上启动一次去某处会话。
          
      en: >
          Starts a goto session on the incremental map.
          
  - protocol: file
    path: "backend/nav_mapping.py#NavSession.on_map_update"
    description:
      zh: >
          在地图更新后重新规划。
          
      en: >
          Reacts to a new map update by re-planning.
          
---
