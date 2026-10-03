---
uid: "9e500015"
id: neko-vrc-body.backend.navigator.wander.direction-memory.record-outcome
parent: neko-vrc-body.backend.navigator.wander.direction-memory
name: {zh: "方向结果记录", en: "Direction Outcome Recording"}
description:
  zh: >
      把每段闲逛执行的实测结果写进方向记忆。
      
  en: >
      Writes the measured outcome of each wander execution into the direction memory.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.685Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 955
    end_line: 1154
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._record_direction_memory_locked"
    description:
      zh: >
          把一段闲逛执行的实测结果写进方向记忆。
          
      en: >
          Writes the measured outcome of a wander leg into the direction memory.
          
---
