---
uid: "9e420002"
id: neko-vrc-body.backend.navigator.goal-check
parent: neko-vrc-body.backend.navigator
name: {zh: "目标可达性检查", en: "Goal reachability check"}
description:
  zh: >
      每个控制拍的入口：先判定目标是否仍然可达，再产生决策。
      
  en: >
      Entry point of every control tick: first decide whether the goal is still reachable, then produce a decision.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.674Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 532
    end_line: 668
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator.tick"
    description:
      zh: >
          推进一个控制拍并返回本拍决策。
          
      en: >
          Advances one control tick and returns that tick's decision.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._pre_tick_goal_check"
    description:
      zh: >
          控制拍前的目标可达性前置判定。
          
      en: >
          Pre-tick reachability check for the current goal.
          
---
