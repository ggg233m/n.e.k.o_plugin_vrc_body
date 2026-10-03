---
uid: "9e400009"
id: neko-vrc-body.backend.perception.frame-sources.mss
parent: neko-vrc-body.backend.perception.frame-sources
name: {zh: "MSS 采集源", en: "MSS Frame Source"}
description:
  zh: >
      纯 mss 桌面采集器，延迟导入并可在多个后端候选之间切换。
      
  en: >
      A pure mss desktop capture source, imported lazily and able to switch between several backend candidates.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.700Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 572
    end_line: 727
apis:
  - protocol: file
    path: "backend/vision.py#MssFrameSource.read"
    description:
      zh: >
          抓取桌面区域的一帧，未就绪时返回空结果而不抛错。
          
      en: >
          Grabs one frame of the desktop region, returning an empty result rather than raising when not ready.
          
  - protocol: file
    path: "backend/vision.py#MssFrameSource.status"
    description:
      zh: >
          报告当前生效的后端候选、区域尺寸与错误信息。
          
      en: >
          Reports the active backend candidate, region size, and any error.
          
---
