---
uid: "9e410014"
id: neko-vrc-body.backend.traversability.flow
parent: neko-vrc-body.backend.traversability
name: {zh: "光流适配", en: "Optical Flow Adapters"}
description:
  zh: >
      稀疏块 Lucas-Kanade 光流与 OpenCV 稠密光流适配，纹理不足时返回 unknown。
      
  en: >
      Sparse-block Lucas-Kanade flow plus an OpenCV dense-flow adapter, returning unknown when texture is insufficient.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.765Z"
fingerprint: 11dcc0ec1c631d57ef36313f26eea34f84d670596c405513242603922b969415
source:
  - path: "backend/traversability.py"
    line: 93
    end_line: 264
apis:
  - protocol: file
    path: "backend/traversability.py#_lucas_kanade_flow"
    description:
      zh: >
          用稀疏块 Lucas-Kanade 在相邻帧之间估计光流。
          
      en: >
          Estimates optical flow between consecutive frames using sparse-block Lucas-Kanade.
          
  - protocol: file
    path: "backend/traversability.py#_opencv_flow"
    description:
      zh: >
          调用 OpenCV 的稠密光流实现作为替代路径。
          
      en: >
          Calls the OpenCV dense optical-flow implementation as an alternative path.
          
  - protocol: file
    path: "backend/traversability.py#_unknown"
    description:
      zh: >
          构造表示"无法判断"的占位光流结果。
          
      en: >
          Builds the placeholder flow result meaning "cannot be determined".
          
---
