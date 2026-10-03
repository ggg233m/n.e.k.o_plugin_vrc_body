---
uid: 9e3a0314
id: neko-vrc-body.body-kernel.motion.gesture
parent: neko-vrc-body.body-kernel.motion
name: {zh: "手势库", en: "Gesture Library"}
description:
  zh: >
      命名手势库：各手势时长，以及在给定进度上产生的那一帧。
      
  en: >
      The named gesture library: per-gesture durations and the frame each gesture produces at a given progress.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.786Z"
fingerprint: 87b9c47e51e1edf49546d81a698b1088f6f5798e561623d5a8cbfbb63a6e69be
source:
  - path: "motion.py"
    line: 361
    end_line: 516
  - path: "motion.py"
    line: 22
    end_line: 36
apis:
  - protocol: file
    path: "motion.py#gesture_frame"
    description:
      zh: >
          在给定进度上采样一个命名手势。
          
      en: >
          Sample a named gesture at a given progress.
          
  - protocol: file
    path: "motion.py#GESTURE_DURATIONS"
    description:
      zh: >
          每个命名手势的规范时长。
          
      en: >
          The canonical duration of each named gesture.
          
  - protocol: file
    path: "motion.py#GESTURE_NAMES"
    description:
      zh: >
          工具层会接受的全部手势名。
          
      en: >
          Every gesture name the tool layer accepts.
          
---
