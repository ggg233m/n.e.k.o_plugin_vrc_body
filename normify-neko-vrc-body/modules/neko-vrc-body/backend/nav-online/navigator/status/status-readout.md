---
uid: 9e50001a
id: neko-vrc-body.backend.nav-online.navigator.status.status-readout
parent: neko-vrc-body.backend.nav-online.navigator.status
name: {zh: "状态回读", en: "Status Readout"}
description:
  zh: >
      导航器状态与栅格图像的对外回读。
      
  en: >
      Outward readout of the navigator status and its grid image.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.653Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 1076
    end_line: 1149
apis:
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator.status"
    description:
      zh: >
          导航器状态的对外回读。
          
      en: >
          Outward readout of the navigator status.
          
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator.grid_png"
    description:
      zh: >
          把导航器栅格渲染成 PNG。
          
      en: >
          Renders the navigator grid as a PNG.
          
---
