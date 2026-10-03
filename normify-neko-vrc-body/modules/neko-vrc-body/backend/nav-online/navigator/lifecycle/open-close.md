---
uid: 9e70000d
id: neko-vrc-body.backend.nav-online.navigator.lifecycle.open-close
parent: neko-vrc-body.backend.nav-online.navigator.lifecycle
name: {zh: "开启与关闭", en: "Open and Close"}
description:
  zh: >
      三线程在线导航器的构造、复位与启停。
      
  en: >
      Construction, reset and the start/stop of the three-thread online navigator.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.647Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 510
    end_line: 646
apis:
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator.start"
    description:
      zh: >
          启动三线程在线导航器。
          
      en: >
          Start the three-thread online navigator.
          
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator.stop"
    description:
      zh: >
          停止在线导航器。
          
      en: >
          Stop the online navigator.
          
---
