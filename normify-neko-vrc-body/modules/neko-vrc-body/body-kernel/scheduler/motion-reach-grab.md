---
uid: 9e3a032a
id: neko-vrc-body.body-kernel.scheduler.motion-reach-grab
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "伸手抓取动作", en: "Reach and Grab Motion"}
description:
  zh: >
      启动普通目标姿态过渡，以及伸手/合手/收回的三相位序列。
      
  en: >
      Starting a plain target-pose transition and the three-phase reach, close and return sequence.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.797Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 1043
    end_line: 1104
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._start_target_motion"
    description:
      zh: >
          启动一个目标姿态动作，可带完成回调。
          
      en: >
          Start a target-pose motion with an optional completion callback.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._start_reach_and_grab"
    description:
      zh: >
          启动伸手、合手、收回的相位序列。
          
      en: >
          Start a reach, close and return phase sequence.
          
---
