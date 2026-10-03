---
uid: 9e40000b
id: neko-vrc-body.backend.perception.frame-sources.wgc
parent: neko-vrc-body.backend.perception.frame-sources
name: {zh: "WGC 窗口采集源", en: "WGC Window Frame Source"}
description:
  zh: >
      按窗口捕获的采集源，被其它窗口遮挡时内容不受影响。
      
  en: >
      A per-window capture source whose content is unaffected when other windows occlude it.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.701Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 1056
    end_line: 1267
apis:
  - protocol: file
    path: "backend/vision.py#WgcWindowFrameSource.read"
    description:
      zh: >
          从 WGC 会话读取目标窗口的一帧画面。
          
      en: >
          Reads one frame of the target window from the WGC session.
          
  - protocol: file
    path: "backend/vision.py#WgcWindowFrameSource._reopen"
    description:
      zh: >
          在窗口句柄或尺寸变化后重建捕获会话。
          
      en: >
          Rebuilds the capture session after the window handle or size changes.
          
  - protocol: file
    path: "backend/vision.py#WgcWindowFrameSource._is_minimized"
    description:
      zh: >
          判断 WGC 目标窗口当前是否已最小化。
          
      en: >
          Determines whether the WGC target window is currently minimized.
          
---
