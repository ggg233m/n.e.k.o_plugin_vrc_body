---
uid: "9e400013"
id: neko-vrc-body.backend.perception.semantic.descriptor
parent: neko-vrc-body.backend.perception.semantic
name: {zh: "视角指纹", en: "Viewpoint Fingerprint"}
description:
  zh: >
      低分辨率背景指纹，只在外观歧义时用来判断摄像机是否还在同一视角。
      
  en: >
      A low-resolution background fingerprint used only when appearance is ambiguous, to tell whether the camera is still at the same viewpoint.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.715Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 1913
    end_line: 1933
apis:
  - protocol: file
    path: "backend/vision.py#_semantic_descriptor"
    description:
      zh: >
          把一帧压缩成低分辨率背景指纹。
          
      en: >
          Compresses one frame into a low-resolution background fingerprint.
          
  - protocol: file
    path: "backend/vision.py#_semantic_similarity"
    description:
      zh: >
          比较两个背景指纹的相似度，判断视角是否未变。
          
      en: >
          Compares the similarity of two background fingerprints to tell whether the viewpoint is unchanged.
          
---
