---
uid: "9e430011"
id: neko-vrc-body.backend.nav-xsession.tracker.keyframe
parent: neko-vrc-body.backend.nav-xsession.tracker
name: {zh: "关键帧查询与验证", en: "Keyframe Query & Verification"}
description:
  zh: >
      每个关键帧查询并几何验证，产出跨会话约束候选。
      
  en: >
      Queries and geometrically verifies every keyframe, producing cross-session constraint candidates.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.660Z"
fingerprint: bb74d4fa731d16faee2a784a6247828d8c29ded78a95d4fda7cd69a1ce17ddba
source:
  - path: "backend/nav_xsession.py"
    line: 714
    end_line: 801
apis:
  - protocol: file
    path: "backend/nav_xsession.py#XSessionTracker.on_keyframe"
    description:
      zh: >
          用新关键帧查询并生成约束候选。
          
      en: >
          Queries with a new keyframe and generates constraint candidates.
          
  - protocol: file
    path: "backend/nav_xsession.py#XSessionTracker._verify_one"
    description:
      zh: >
          对单个候选做几何验证。
          
      en: >
          Geometrically verifies one candidate.
          
---
