---
uid: 9e40000d
id: neko-vrc-body.backend.perception.frame-sources.window-tracked
parent: neko-vrc-body.backend.perception.frame-sources
name: {zh: "窗口跟踪采集源", en: "Window-Tracked Frame Source"}
description:
  zh: >
      按 TTL 重新解析窗口矩形，窗口移动或改分辨率后重建内部采集源。
      
  en: >
      Re-resolves the window rectangle on a TTL, rebuilding the inner capture source after the window moves or changes resolution.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.702Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 1340
    end_line: 1510
apis:
  - protocol: file
    path: "backend/vision.py#WindowTrackedFrameSource._refresh"
    description:
      zh: >
          按 TTL 重新解析窗口矩形，必要时重建内部采集源。
          
      en: >
          Re-resolves the window rectangle on a TTL, rebuilding the inner capture source when needed.
          
  - protocol: file
    path: "backend/vision.py#WindowTrackedFrameSource._probe_visibility"
    description:
      zh: >
          探测目标窗口当前是否可见，以决定继续采集还是暂停。
          
      en: >
          Probes whether the target window is currently visible to decide between capturing and pausing.
          
  - protocol: file
    path: "backend/vision.py#WindowTrackedFrameSource.read"
    description:
      zh: >
          从当前跟踪到的窗口采集一帧画面。
          
      en: >
          Captures one frame from the currently tracked window.
          
---
