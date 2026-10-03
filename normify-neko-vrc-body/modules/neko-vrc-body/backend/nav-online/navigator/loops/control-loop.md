---
uid: 9e50000e
id: neko-vrc-body.backend.nav-online.navigator.loops.control-loop
parent: neko-vrc-body.backend.nav-online.navigator.loops
name: {zh: "控制线程", en: "Control Loop"}
description:
  zh: >
      十赫兹控制线程与每一拍的推进。
      
  en: >
      10 Hz control thread and the advance of each tick.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.648Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 1000
    end_line: 1047
apis:
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator._control_loop"
    description:
      zh: >
          十赫兹控制线程主体。
          
      en: >
          Body of the 10 Hz control thread.
          
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator._control_tick"
    description:
      zh: >
          推进一拍控制。
          
      en: >
          Advances one control tick.
          
---
