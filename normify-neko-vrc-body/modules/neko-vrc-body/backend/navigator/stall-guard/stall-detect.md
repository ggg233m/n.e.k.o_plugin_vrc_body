---
uid: 9e50000f
id: neko-vrc-body.backend.navigator.stall-guard.stall-detect
parent: neko-vrc-body.backend.navigator.stall-guard
name: {zh: "卡死检测", en: "Stall Detection"}
description:
  zh: >
      卡死检测：依据位移、转向与受阻证据判定无进展。
      
  en: >
      Stall detection: judges lack of progress from displacement, turning and blocked evidence.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.681Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 1388
    end_line: 1618
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._stall_guard"
    description:
      zh: >
          依据位移、转向与受阻证据判定导航器是否卡死。
          
      en: >
          Decides whether the navigator is stalled from displacement, turning and blocked evidence.
          
---
