---
uid: 9e41000a
id: neko-vrc-body.backend.local-perception.detector.observe
parent: neko-vrc-body.backend.local-perception.detector
name: {zh: "观测生成", en: "Observation Publishing"}
description:
  zh: >
      把检测结果转成世界观测批次，附上类别、置信度与来源。
      
  en: >
      Turns detections into a batch of world observations carrying class, confidence and provenance.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.611Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 1536
    end_line: 1699
apis:
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector.observe"
    description:
      zh: >
          对当前帧执行推理并输出世界观测批次。
          
      en: >
          Runs inference on the current frame and emits a batch of world observations.
          
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._entities"
    description:
      zh: >
          从检测结果整理出带类别、置信度与来源的实体列表。
          
      en: >
          Organizes detections into an entity list with class, confidence and provenance.
          
---
