---
uid: 9e50000d
id: neko-vrc-body.backend.nav-online.navigator.loops.mapping-loop
parent: neko-vrc-body.backend.nav-online.navigator.loops
name: {zh: "建图线程", en: "Mapping Loop"}
description:
  zh: >
      建图线程与地图更新请求。
      
  en: >
      Mapping thread and the map-update requests it issues.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.650Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 897
    end_line: 997
apis:
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator._mapping_loop"
    description:
      zh: >
          建图线程主体。
          
      en: >
          Mapping thread body.
          
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator._request_map_update"
    description:
      zh: >
          请求一次地图更新。
          
      en: >
          Requests one map update.
          
---
