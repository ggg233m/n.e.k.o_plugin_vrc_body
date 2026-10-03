---
uid: "9e410007"
id: neko-vrc-body.backend.local-perception.detector.preprocess
parent: neko-vrc-body.backend.local-perception.detector
name: {zh: "前处理", en: "Preprocessing"}
description:
  zh: >
      帧转数组、双线性缩放与各后端的前处理。
      
  en: >
      Frame-to-array conversion, bilinear scaling and per-backend preprocessing.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.613Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 973
    end_line: 1138
apis:
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._preprocess"
    description:
      zh: >
          把原始帧转换成当前后端所需的张量输入。
          
      en: >
          Converts a raw frame into the tensor input the active backend expects.
          
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._resize_bilinear"
    description:
      zh: >
          用双线性插值把帧缩放到模型输入尺寸。
          
      en: >
          Resizes a frame to the model input size with bilinear interpolation.
          
---
