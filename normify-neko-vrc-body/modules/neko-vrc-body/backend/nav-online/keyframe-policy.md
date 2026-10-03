---
uid: "9e420015"
id: neko-vrc-body.backend.nav-online.keyframe-policy
parent: neko-vrc-body.backend.nav-online
name: {zh: "关键帧策略", en: "Keyframe policy"}
description:
  zh: >
      双目帧要不要成为关键帧：跳过、追加，或原地补帧。
      
  en: >
      Whether a stereo frame becomes a keyframe: skip it, append it, or backfill in place.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.644Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 130
    end_line: 163
apis:
  - protocol: file
    path: "backend/nav_online.py#KeyframePolicy.decide"
    description:
      zh: >
          判定一个双目帧是否成为关键帧。
          
      en: >
          Decides whether a stereo frame becomes a keyframe.
          
---
