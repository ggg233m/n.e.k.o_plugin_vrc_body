---
uid: "9e420007"
id: neko-vrc-body.backend.navigator.goal-completion
parent: neko-vrc-body.backend.navigator
name: {zh: "目标达成通知", en: "Goal completion notice"}
description:
  zh: >
      目标达成时通知上层，并清掉与目标绑定的状态。
      
  en: >
      Notifies the upper layer when the goal is reached and clears the state bound to that goal.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.675Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 1208
    end_line: 1231
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._notify_goal_complete"
    description:
      zh: >
          向上层播报目标达成并清理目标状态。
          
      en: >
          Reports goal completion upward and clears the goal state.
          
---
