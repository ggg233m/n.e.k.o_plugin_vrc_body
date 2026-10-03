---
uid: 9e3a0025
id: neko-vrc-body.config.vision-section
parent: neko-vrc-body.config
name: {zh: "视觉配置段", en: "Vision Section"}
description:
  zh: >
      采集后端选择、检测节奏、拉帧预算与感知 worker 的 VLM 路由。
      
  en: >
      Capture backend choice, detector cadence, frame budget and VLM route for the perception worker.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.815Z"
fingerprint: 55757bfaa420ee4e8c4479614d38026395bc79552cfe74aec0d249ef070ee23f
source:
  - path: "config.py"
    line: 358
    end_line: 446
apis:
  - protocol: file
    path: "config.py#VisionConfig"
    description:
      zh: >
          与模型无关的感知 worker 配置；具体检测器由后端注入。
          
      en: >
          Model-independent perception worker settings; the concrete detector is injected by the backend.
          
---
