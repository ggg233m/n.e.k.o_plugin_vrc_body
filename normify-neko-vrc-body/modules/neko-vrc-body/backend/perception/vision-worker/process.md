---
uid: "9e400018"
id: neko-vrc-body.backend.perception.vision-worker.process
parent: neko-vrc-body.backend.perception.vision-worker
name: {zh: "推理循环", en: "Inference Loop"}
description:
  zh: >
      推理循环：检测、地面可见范围与可通行性预测的调度。
      
  en: >
      The inference loop: scheduling of detection, visible ground extent, and traversability prediction.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.722Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 2533
    end_line: 2729
apis:
  - protocol: file
    path: "backend/vision.py#VisionWorker._process_loop"
    description:
      zh: >
          推理循环：按节流取帧、跑检测并发布观测。
          
      en: >
          The inference loop: consumes frames under throttling, runs detection, and publishes observations.
          
  - protocol: file
    path: "backend/vision.py#VisionWorker._estimate_traversability"
    description:
      zh: >
          由地面可见比例与坡度线索推断可通行区域。
          
      en: >
          Infers traversable area from visible ground coverage and slope cues.
          
---
