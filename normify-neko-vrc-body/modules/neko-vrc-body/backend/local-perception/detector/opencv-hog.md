---
uid: "9e410006"
id: neko-vrc-body.backend.local-perception.detector.opencv-hog
parent: neko-vrc-body.backend.local-perception.detector
name: {zh: "OpenCV 与 HOG 后端", en: "OpenCV and HOG Backends"}
description:
  zh: >
      OpenCV DNN 与 HOG 两个轻量后端，作为重模型不可用时的兜底。
      
  en: >
      Two lightweight backends, OpenCV DNN and HOG, that act as fallbacks when the heavy model is unavailable.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.612Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 907
    end_line: 970
apis:
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._initialize_opencv_dnn"
    description:
      zh: >
          加载 OpenCV DNN 轻量模型作为兜底后端。
          
      en: >
          Loads a lightweight OpenCV DNN model as the fallback backend.
          
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._initialize_hog"
    description:
      zh: >
          初始化 HOG 行人检测器作为最后一级兜底。
          
      en: >
          Initializes the HOG pedestrian detector as the last-resort fallback.
          
---
