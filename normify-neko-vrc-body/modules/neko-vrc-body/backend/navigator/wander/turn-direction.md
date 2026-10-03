---
uid: "9e420004"
id: neko-vrc-body.backend.navigator.wander.turn-direction
parent: neko-vrc-body.backend.navigator.wander
name: {zh: "转向方向合约", en: "Turn direction contract"}
description:
  zh: >
      依据方向合约决定转向方向，并采样上一次输出状态作为起点。
      
  en: >
      Decides the turn direction from the direction contract and samples the previous output state as the starting point.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.688Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 662
    end_line: 722
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._turn_direction_by_contract"
    description:
      zh: >
          依据方向合约决定本拍的转向方向。
          
      en: >
          Decides the turn direction from the direction contract.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._free_roam_params"
    description:
      zh: >
          推导本拍自由漫游所用的参数。
          
      en: >
          Derives the free roam parameters for this tick.
          
---
