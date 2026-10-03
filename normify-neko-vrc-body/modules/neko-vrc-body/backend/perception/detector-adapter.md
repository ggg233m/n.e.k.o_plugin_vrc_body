---
uid: "9e420013"
id: neko-vrc-body.backend.perception.detector-adapter
parent: neko-vrc-body.backend.perception
name: {zh: "检测器适配接口", en: "Detector Adapter"}
description:
  zh: >
      面向 YOLOX 与深度模型包的安全适配接口；模型或运行时缺失时如实回报，而不是编造检测结果。
      
  en: >
      A safe adapter interface for YOLOX and depth model packages; when the model or runtime is missing it reports that rather than inventing detections.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.695Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 1513
    end_line: 1570
apis:
  - protocol: file
    path: "backend/vision.py#OpenVinoLocalDetector.observe"
    description:
      zh: >
          运行检测器并返回一批世界观测。
          
      en: >
          Run the detector and return a world observation batch.
          
  - protocol: file
    path: "backend/vision.py#OpenVinoLocalDetector.status"
    description:
      zh: >
          报告模型加载状态，绝不抛异常。
          
      en: >
          Report model load state without ever raising.
          
deps:
  - kind: call
    to: neko-vrc-body.backend.local-perception.detector.observe
    label: {zh: "解码并跟踪", en: "Decode and track"}
---
