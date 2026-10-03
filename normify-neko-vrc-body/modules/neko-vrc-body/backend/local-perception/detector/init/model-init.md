---
uid: "9e500000"
id: neko-vrc-body.backend.local-perception.detector.init.model-init
parent: neko-vrc-body.backend.local-perception.detector.init
name: {zh: "检测器构造", en: "Detector Construction"}
description:
  zh: >
      构造检测器：记录配置、候选模型路径、设备选择与线程上限。
      
  en: >
      Constructs the detector: stores the configuration, candidate model paths, device selection and thread caps.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.609Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 382
    end_line: 552
apis:
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector.__init__"
    description:
      zh: >
          记录配置、候选模型路径、设备选择与线程上限。
          
      en: >
          Stores the configuration, candidate model paths, device selection and thread caps.
          
---
