---
uid: "9e400017"
id: neko-vrc-body.backend.perception.vision-worker.capture
parent: neko-vrc-body.backend.perception.vision-worker
name: {zh: "采集循环", en: "Capture Loop"}
description:
  zh: >
      有界采集循环：丢帧优先于堆积，且不触碰身体调度线程。
      
  en: >
      A bounded capture loop that drops frames rather than piling them up and never touches the body scheduling thread.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.720Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 2332
    end_line: 2531
apis:
  - protocol: file
    path: "backend/vision.py#VisionWorker.start"
    description:
      zh: >
          启动采集与推理两个线程，进入运行态。
          
      en: >
          Starts the capture and inference threads and enters the running state.
          
  - protocol: file
    path: "backend/vision.py#VisionWorker._capture_loop"
    description:
      zh: >
          有界采集循环：保持最新一帧，忙时丢帧而不堆积。
          
      en: >
          The bounded capture loop: keeps the newest frame and drops frames instead of piling them up when busy.
          
  - protocol: file
    path: "backend/vision.py#VisionWorker._obscured"
    description:
      zh: >
          判定目标画面此刻是否被遮挡或不可见，输出遮挡态观测。
          
      en: >
          Decides whether the target view is currently occluded or invisible and emits an obscured observation.
          
---
