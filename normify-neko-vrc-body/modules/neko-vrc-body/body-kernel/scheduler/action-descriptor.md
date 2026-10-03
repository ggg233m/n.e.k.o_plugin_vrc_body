---
uid: 9e3a032e
id: neko-vrc-body.body-kernel.scheduler.action-descriptor
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "动作描述符", en: "Action Descriptor"}
description:
  zh: >
      工具回读的动作描述符：正在跑什么、跑了多久、结果如何。
      
  en: >
      The action descriptor the tools read back: what is running, for how long, and with what outcome.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.790Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 1402
    end_line: 1463
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._action_descriptor"
    description:
      zh: >
          为状态回读汇总当前动作。
          
      en: >
          Summarise the active action for the status read-out.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._set_current_action"
    description:
      zh: >
          把新启动的动作发布为当前动作。
          
      en: >
          Publish a newly started action as current.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._clear_current_action"
    description:
      zh: >
          清空当前动作并记录结果。
          
      en: >
          Clear the current action with a recorded outcome.
          
---
