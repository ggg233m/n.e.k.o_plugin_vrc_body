---
uid: 9e70000e
id: neko-vrc-body.backend.nav-online.navigator.lifecycle.session-hooks
parent: neko-vrc-body.backend.nav-online.navigator.lifecycle
name: {zh: "会话接线", en: "Session Hooks"}
description:
  zh: >
      把跨会话追踪器与 navmesh 记忆写入器接到导航器的会话边界上。
      
  en: >
      Wiring the cross-session tracker and the navmesh memory writer to the navigator's session boundaries.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.648Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 649
    end_line: 738
apis:
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator._begin_memory"
    description:
      zh: >
          打开 navmesh 记忆写入器。
          
      en: >
          Open the navmesh memory writer.
          
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator._align_xsession"
    description:
      zh: >
          会话结束时采纳过往会话的世界系。
          
      en: >
          Adopt a past session's world frame at session end.
          
---
