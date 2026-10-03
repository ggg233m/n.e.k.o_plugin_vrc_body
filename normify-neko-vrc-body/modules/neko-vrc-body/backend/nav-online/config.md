---
uid: "9e420014"
id: neko-vrc-body.backend.nav-online.config
parent: neko-vrc-body.backend.nav-online
name: {zh: "在线导航配置与基线", en: "Online navigation config and baseline"}
description:
  zh: >
      在线建图的频率、阈值与基线校验，以及 HMD 到底部坐标系的旋转。
      
  en: >
      Frequencies, thresholds, and baseline checks for online mapping, plus the rotation from HMD to the base coordinate system.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.642Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 38
    end_line: 127
apis:
  - protocol: file
    path: "backend/nav_online.py#OnlineNavConfig"
    description:
      zh: >
          在线建图与导航的可调参数。
          
      en: >
          Tunable parameters of online mapping and navigation.
          
  - protocol: file
    path: "backend/nav_online.py#hmd_to_base_rotation"
    description:
      zh: >
          把 HMD 朝向旋转到底部坐标系。
          
      en: >
          Rotates an HMD orientation into the base coordinate system.
          
  - protocol: file
    path: "backend/nav_online.py#check_baseline"
    description:
      zh: >
          按配置的期望校验实测基线。
          
      en: >
          Validates a measured baseline against the configured expectations.
          
---
