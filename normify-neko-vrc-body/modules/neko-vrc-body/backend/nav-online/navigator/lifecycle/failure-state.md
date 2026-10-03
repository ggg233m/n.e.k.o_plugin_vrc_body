---
uid: 9e70000f
id: neko-vrc-body.backend.nav-online.navigator.lifecycle.failure-state
parent: neko-vrc-body.backend.nav-online.navigator.lifecycle
name: {zh: "失败状态", en: "Failure State"}
description:
  zh: >
      失败与停止路径，以及读取当前融合位姿估计的唯一入口。
      
  en: >
      The failure and halt paths, and the single place the current fused pose estimate is read.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.646Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 740
    end_line: 761
apis:
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator._fail"
    description:
      zh: >
          记录失败并停止它所触及的一切。
          
      en: >
          Record a failure and stop everything it touched.
          
  - protocol: file
    path: "backend/nav_online.py#OnlineNavigator._est"
    description:
      zh: >
          读取当前融合位姿估计。
          
      en: >
          Read the current fused pose estimate.
          
---
