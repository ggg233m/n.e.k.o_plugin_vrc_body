---
uid: 9e43000d
id: neko-vrc-body.backend.nav-xsession.gauge.elastic
parent: neko-vrc-body.backend.nav-xsession.gauge
name: {zh: "弹性锚定位姿图", en: "Elastic Anchor Pose Graph"}
description:
  zh: >
      弹性锚定位姿图：节点只含 x、y、yaw，里程边保形、跨会话锚为绝对观测。
      
  en: >
      An elastic anchor pose graph whose nodes hold only x, y and yaw, with shape-preserving odometry edges and absolute cross-session anchor observations.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.656Z"
fingerprint: bb74d4fa731d16faee2a784a6247828d8c29ded78a95d4fda7cd69a1ce17ddba
source:
  - path: "backend/nav_xsession.py"
    line: 416
    end_line: 505
apis:
  - protocol: file
    path: "backend/nav_xsession.py#_elastic_merge"
    description:
      zh: >
          在锚定观测之间弹性调整各会话位姿。
          
      en: >
          Elastically adjusts the per-session poses against the anchor observations.
          
---
