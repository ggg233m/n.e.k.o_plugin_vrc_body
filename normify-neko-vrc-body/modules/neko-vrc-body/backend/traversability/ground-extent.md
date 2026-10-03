---
uid: "9e410016"
id: neko-vrc-body.backend.traversability.ground-extent
parent: neko-vrc-body.backend.traversability
name: {zh: "地面可见范围估计", en: "Ground Extent Estimation"}
description:
  zh: >
      单帧地面可见范围估计，补上光流在站着不动时的盲区，输出是序数而非米制。
      
  en: >
      Single-frame ground visible-extent estimation that covers the blind spot optical flow has when standing still, returning ordinal rather than metric output.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.765Z"
fingerprint: 11dcc0ec1c631d57ef36313f26eea34f84d670596c405513242603922b969415
source:
  - path: "backend/traversability.py"
    line: 445
    end_line: 615
apis:
  - protocol: file
    path: "backend/traversability.py#GroundExtentEstimator.estimate"
    description:
      zh: >
          从单帧估计地面可见范围的序数等级。
          
      en: >
          Estimates the ordinal level of ground visible extent from a single frame.
          
---
