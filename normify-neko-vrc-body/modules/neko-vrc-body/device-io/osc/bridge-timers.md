---
uid: 9e3a0347
id: neko-vrc-body.device-io.osc.bridge-timers
parent: neko-vrc-body.device-io.osc
name: {zh: "定时循环", en: "Timer Loops"}
description:
  zh: >
      两个后台维护循环：让定时轴过期，以及触发到期的调度输入。
      
  en: >
      The two background maintenance loops: expiring timed axes and firing due scheduled inputs.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.828Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 717
    end_line: 764
apis:
  - protocol: file
    path: "osc.py#VrchatOscBridge._run_axis_expirations"
    description:
      zh: >
          让到期轴失效并停止它们。
          
      en: >
          Expire timed-out axes and stop them.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge._run_due_inputs"
    description:
      zh: >
          触发所有截止时间已到的调度输入。
          
      en: >
          Fire every scheduled input whose deadline has passed.
          
---
