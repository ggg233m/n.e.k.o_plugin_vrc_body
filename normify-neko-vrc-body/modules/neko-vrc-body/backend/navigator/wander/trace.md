---
uid: "9e420005"
id: neko-vrc-body.backend.navigator.wander.trace
parent: neko-vrc-body.backend.navigator.wander
name: {zh: "闲逛执行轨迹", en: "Wander execution trace"}
description:
  zh: >
      闲逛执行的逐步记录与汇总，供方向记忆消费。
      
  en: >
      Step-by-step recording and summary of a wander execution, consumed by the direction memory.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.687Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 724
    end_line: 953
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._start_wander_trace_locked"
    description:
      zh: >
          在持锁状态下开始一段新的闲逛执行轨迹。
          
      en: >
          Opens a new wander execution trace while holding the lock.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._record_wander_execution"
    description:
      zh: >
          把一次闲逛执行写入轨迹记录。
          
      en: >
          Records one wander step into the execution trace.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator._wander_execution_summary_locked"
    description:
      zh: >
          在持锁状态下汇总本次闲逛执行。
          
      en: >
          Summarizes the wander execution while holding the lock.
          
---
