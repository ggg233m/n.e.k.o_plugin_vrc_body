---
uid: "9e500014"
id: neko-vrc-body.backend.local-perception.detector.decode.row-decode
parent: neko-vrc-body.backend.local-perception.detector.decode
name: {zh: "行解码", en: "Row Decoding"}
description:
  zh: >
      把行解码成检测结果。
      
  en: >
      Turns rows into detections.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.607Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 1413
    end_line: 1534
apis:
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._decode_rows"
    description:
      zh: >
          把归一后的行解码成检测结果。
          
      en: >
          Turns normalised rows into detections.
          
---
