---
uid: "9e410015"
id: neko-vrc-body.backend.traversability.optical-flow
parent: neko-vrc-body.backend.traversability
name: {zh: "光流可通行性预测", en: "Optical-Flow Traversability"}
description:
  zh: >
      连续帧局部可通行性预测器，不选方向也不发送控制器输入。
      
  en: >
      A local traversability predictor over consecutive frames that selects no direction and emits no controller input.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.766Z"
fingerprint: 11dcc0ec1c631d57ef36313f26eea34f84d670596c405513242603922b969415
source:
  - path: "backend/traversability.py"
    line: 267
    end_line: 442
apis:
  - protocol: file
    path: "backend/traversability.py#OpticalFlowTraversability.estimate"
    description:
      zh: >
          基于光流给出局部可通行性判据与风险等级。
          
      en: >
          Derives local traversability cues and a risk level from the optical flow.
          
  - protocol: file
    path: "backend/traversability.py#OpticalFlowTraversability.dependency_available"
    description:
      zh: >
          报告光流后端依赖是否可用，不可用时明确降级。
          
      en: >
          Reports whether the optical-flow backend dependency is available, degrading explicitly when it is not.
          
---
