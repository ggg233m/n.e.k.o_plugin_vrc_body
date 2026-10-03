---
uid: "9e420009"
id: neko-vrc-body.backend.navigator.behavior.phase
parent: neko-vrc-body.backend.navigator.behavior
name: {zh: "行为阶段", en: "Behavior phase"}
description:
  zh: >
      站立不动的决策，以及行为阶段的计时。
      
  en: >
      The decision to stay stationary and the timing of behavior phases.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.664Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 1234
    end_line: 1265
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._stationary_decision"
    description:
      zh: >
          构造原地站立不动的决策。
          
      en: >
          Builds the decision for standing still without moving.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._set_behavior_phase"
    description:
      zh: >
          切换到某个行为阶段并启动其计时。
          
      en: >
          Switches to a behavior phase and starts its timer.
          
---
