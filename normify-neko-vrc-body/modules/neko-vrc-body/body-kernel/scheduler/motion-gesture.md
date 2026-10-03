---
uid: 9e3a032b
id: neko-vrc-body.body-kernel.scheduler.motion-gesture
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "手势动作", en: "Gesture Motion"}
description:
  zh: >
      启动命名手势，含在允许其驱动身体之前执行的轨迹校验。
      
  en: >
      Starting a named gesture, including the trajectory validation that runs before it is allowed to move the body.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.797Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 1106
    end_line: 1179
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._start_gesture"
    description:
      zh: >
          启动一个命名手势动作。
          
      en: >
          Start a named gesture motion.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._validate_gesture_trajectory"
    description:
      zh: >
          在允许手势驱动身体之前校验其轨迹。
          
      en: >
          Validate a gesture trajectory before it is allowed to move the body.
          
deps:
  - kind: call
    to: neko-vrc-body.body-kernel.motion.gesture
    label: {zh: "采样命名手势", en: "Sample the named gesture"}
  - kind: call
    to: neko-vrc-body.body-kernel.motion.interpolation
    label: {zh: "在姿态之间插值", en: "Interpolate between poses"}
---
