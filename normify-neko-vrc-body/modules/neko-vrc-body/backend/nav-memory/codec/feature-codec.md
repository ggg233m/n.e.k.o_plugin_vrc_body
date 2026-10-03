---
uid: 9e70000a
id: neko-vrc-body.backend.nav-memory.codec.feature-codec
parent: neko-vrc-body.backend.nav-memory.codec
name: {zh: "特征编解码", en: "Feature Codec"}
description:
  zh: >
      单个关键帧地图侧 ORB 特征的二进制编解码。
      
  en: >
      Binary encoding and decoding of one keyframe's map-side ORB features.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.637Z"
fingerprint: 8ac3ac2937f0825e7d07c1267ffa2e59000b81e0743552690708cb1b776bbb0e
source:
  - path: "backend/nav_memory.py"
    line: 94
    end_line: 111
apis:
  - protocol: file
    path: "backend/nav_memory.py#encode_features"
    description:
      zh: >
          编码一帧的地图侧特征。
          
      en: >
          Encode one frame's map-side features.
          
  - protocol: file
    path: "backend/nav_memory.py#load_features"
    description:
      zh: >
          把一帧特征解码回 float32。
          
      en: >
          Decode a frame's features back to float32.
          
---
