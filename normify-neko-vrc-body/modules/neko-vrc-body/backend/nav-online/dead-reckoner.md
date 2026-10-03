---
uid: "9e420017"
id: neko-vrc-body.backend.nav-online.dead-reckoner
parent: neko-vrc-body.backend.nav-online
name: {zh: "航位推算", en: "Dead reckoning"}
description:
  zh: >
      本地速度与 HMD 朝向积分成地图系位姿，样本之间按前值外推。
      
  en: >
      Integrates local speed and HMD heading into a map-frame pose and extrapolates with the previous value between samples.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.643Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 273
    end_line: 401
apis:
  - protocol: file
    path: "backend/nav_online.py#DeadReckoner.update"
    description:
      zh: >
          把新的运动样本积分进当前位姿。
          
      en: >
          Folds a new motion sample into the integrated pose.
          
  - protocol: file
    path: "backend/nav_online.py#DeadReckoner.predict"
    description:
      zh: >
          在样本之间用前值外推位姿。
          
      en: >
          Extrapolates the pose between samples with the previous value.
          
---
