---
uid: "9e500013"
id: neko-vrc-body.backend.local-perception.detector.decode.nms
parent: neko-vrc-body.backend.local-perception.detector.decode
name: {zh: "重叠框抑制", en: "Overlap Suppression"}
description:
  zh: >
      重叠框抑制与过小框丢弃。
      
  en: >
      Overlap suppression and dropping boxes that are too small.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.606Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 1331
    end_line: 1412
apis:
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._suppress_overlaps"
    description:
      zh: >
          抑制重叠框。
          
      en: >
          Suppresses overlapping boxes.
          
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._is_too_small"
    description:
      zh: >
          判断某个框是否小到不可能为真。
          
      en: >
          Tells whether a box is too small to be real.
          
---
