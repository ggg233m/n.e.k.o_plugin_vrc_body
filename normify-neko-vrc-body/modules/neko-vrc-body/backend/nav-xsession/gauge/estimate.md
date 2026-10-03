---
uid: 9e43000c
id: neko-vrc-body.backend.nav-xsession.gauge.estimate
parent: neko-vrc-body.backend.nav-xsession.gauge
name: {zh: "单次规范估计", en: "One-Shot Gauge Estimate"}
description:
  zh: >
      从跨会话约束估计新会话到旧会话地图系的旋转与平移规范。
      
  en: >
      Estimates the rotation and translation gauge from a new session into an old session's map frame.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.656Z"
fingerprint: bb74d4fa731d16faee2a784a6247828d8c29ded78a95d4fda7cd69a1ce17ddba
source:
  - path: "backend/nav_xsession.py"
    line: 344
    end_line: 413
apis:
  - protocol: file
    path: "backend/nav_xsession.py#estimate_gauge"
    description:
      zh: >
          由对应点集解出旋转与平移规范。
          
      en: >
          Solves the rotation and translation gauge from correspondence sets.
          
---
