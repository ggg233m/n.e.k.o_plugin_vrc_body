---
uid: "9e410013"
id: neko-vrc-body.backend.traversability.config
parent: neko-vrc-body.backend.traversability
name: {zh: "可通行性配置", en: "Traversability Configuration"}
description:
  zh: >
      光流估计与地面可见范围的有界参数，数值是相对风险阈值而非米制距离。
      
  en: >
      Bounded parameters for optical-flow estimation and ground visible extent, where the numbers are relative risk thresholds rather than metric distances.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.764Z"
fingerprint: 11dcc0ec1c631d57ef36313f26eea34f84d670596c405513242603922b969415
source:
  - path: "backend/traversability.py"
    line: 24
    end_line: 90
apis:
  - protocol: file
    path: "backend/traversability.py#TraversabilityConfig"
    description:
      zh: >
          光流可通行性估计的有界参数集合。
          
      en: >
          Bounded parameter set for optical-flow traversability estimation.
          
  - protocol: file
    path: "backend/traversability.py#GroundExtentConfig"
    description:
      zh: >
          地面可见范围估计的有界参数集合。
          
      en: >
          Bounded parameter set for ground visible-extent estimation.
          
---
