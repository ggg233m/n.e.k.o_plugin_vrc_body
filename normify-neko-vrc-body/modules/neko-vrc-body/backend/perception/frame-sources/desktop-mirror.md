---
uid: 9e40000c
id: neko-vrc-body.backend.perception.frame-sources.desktop-mirror
parent: neko-vrc-body.backend.perception.frame-sources
name: {zh: "桌面镜像组合源", en: "Desktop Mirror Composite Source"}
description:
  zh: >
      优先 DXcam、失败回退 MSS 的组合采集器，探测期间两个后端都保持可用。
      
  en: >
      A composite capture source that prefers DXcam and falls back to MSS, keeping both backends usable while probing.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.697Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 1270
    end_line: 1337
apis:
  - protocol: file
    path: "backend/vision.py#DesktopMirrorFrameSource.read"
    description:
      zh: >
          优先从 DXcam 取帧，失败时回退到 MSS 采集。
          
      en: >
          Reads a frame from DXcam first and falls back to MSS capture on failure.
          
  - protocol: file
    path: "backend/vision.py#DesktopMirrorFrameSource.status"
    description:
      zh: >
          报告当前生效的后端以及各候选的可用性与探测结果。
          
      en: >
          Reports the active backend plus each candidate's availability and probe outcome.
          
---
