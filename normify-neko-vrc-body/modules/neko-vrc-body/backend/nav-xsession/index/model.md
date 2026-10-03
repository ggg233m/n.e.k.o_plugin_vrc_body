---
uid: "9e430009"
id: neko-vrc-body.backend.nav-xsession.index.model
parent: neko-vrc-body.backend.nav-xsession.index
name: {zh: "检索索引模型", en: "Retrieval Index Model"}
description:
  zh: >
      世界级只读检索索引，文档即历史会话的关键帧。
      
  en: >
      A world-level read-only retrieval index whose documents are the keyframes of past sessions.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.659Z"
fingerprint: bb74d4fa731d16faee2a784a6247828d8c29ded78a95d4fda7cd69a1ce17ddba
source:
  - path: "backend/nav_xsession.py"
    line: 86
    end_line: 141
apis:
  - protocol: file
    path: "backend/nav_xsession.py#XSessionIndex.query"
    description:
      zh: >
          用特征向量查询最相近的历史关键帧。
          
      en: >
          Queries the nearest historical keyframes for a feature vector.
          
  - protocol: file
    path: "backend/nav_xsession.py#XSessionIndex.load_kf"
    description:
      zh: >
          按需载入某个关键帧的位姿与图像。
          
      en: >
          Loads a keyframe's pose and image on demand.
          
  - protocol: file
    path: "backend/nav_xsession.py#XSessionIndex.doc"
    description:
      zh: >
          取出某个编号文档的向量与元信息。
          
      en: >
          Returns a numbered document's vector and metadata.
          
---
