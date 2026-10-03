---
uid: "9e420012"
id: neko-vrc-body.backend.navigator.turn-gate
parent: neko-vrc-body.backend.navigator
name: {zh: "转向门控", en: "Turn gate"}
description:
  zh: >
      转向门控：避免同一帧修订重复下发转向。
      
  en: >
      Turn gating that avoids re-sending a turn command for the same frame revision.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.683Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 2368
    end_line: 2440
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._send_gated_turn"
    description:
      zh: >
          经过门控后下发转向指令。
          
      en: >
          Sends the turn command after passing the gate.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._turn_gate_reason"
    description:
      zh: >
          给出本次转向被门控抑制的理由。
          
      en: >
          Returns the reason this turn was suppressed by the gate.
          
---
