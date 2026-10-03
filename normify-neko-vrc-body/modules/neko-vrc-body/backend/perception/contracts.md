---
uid: "9e400000"
id: neko-vrc-body.backend.perception.contracts
parent: neko-vrc-body.backend.perception
name: {zh: "感知契约", en: "Perception Contracts"}
description:
  zh: >
      与具体运行时无关的四个小型协议，以及不导入重量级包就能报告可选能力是否存在的探测。
      
  en: >
      Four small runtime-agnostic protocols, plus a probe that reports whether optional capabilities are present without importing heavyweight packages.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.694Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 30
    end_line: 129
apis:
  - protocol: file
    path: "backend/vision.py#VisionObservation"
    description:
      zh: >
          统一的一帧观测结果结构，承载检测、语义与运行状态。
          
      en: >
          The unified per-frame observation result carrying detections, semantics, and runtime state.
          
  - protocol: file
    path: "backend/vision.py#FrameDetector"
    description:
      zh: >
          检测器协议：把一帧图像变成带标签与置信度的框。
          
      en: >
          Detector protocol: turns one image frame into labeled and scored boxes.
          
  - protocol: file
    path: "backend/vision.py#FrameSource"
    description:
      zh: >
          采集源协议：提供帧读取、就绪判断与状态回报。
          
      en: >
          Frame source protocol: provides frame reads, readiness checks, and status reporting.
          
  - protocol: file
    path: "backend/vision.py#SemanticBackend"
    description:
      zh: >
          语义后端协议：给画面附加自然语言场景描述。
          
      en: >
          Semantic backend protocol: attaches natural-language scene descriptions to a frame.
          
  - protocol: file
    path: "backend/vision.py#optional_dependency_status"
    description:
      zh: >
          汇总各可选依赖是否可用的探测结果。
          
      en: >
          Reports the aggregated availability of each optional dependency.
          
---
