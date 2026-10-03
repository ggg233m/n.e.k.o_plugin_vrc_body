---
uid: 9e40001a
id: neko-vrc-body.backend.perception.runtime.init
parent: neko-vrc-body.backend.perception.runtime
name: {zh: "运行时初始化与状态", en: "Runtime Init and State"}
description:
  zh: >
      运行时构造与采集状态、可通行性预测的读写。
      
  en: >
      Runtime construction plus reads and writes of capture state and traversability prediction.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.710Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 2732
    end_line: 2918
apis:
  - protocol: file
    path: "backend/vision.py#VisionRuntime.set_capture_state"
    description:
      zh: >
          设置采集状态（运行/暂停/停止），驱动线程切换。
          
      en: >
          Sets the capture state (running/paused/stopped), driving thread transitions.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime.set_traversability_prediction"
    description:
      zh: >
          更新可通行性预测结果，供导航侧消费。
          
      en: >
          Updates the traversability prediction for consumption on the navigation side.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime.close"
    description:
      zh: >
          停止线程并释放采集源、模型与网络会话。
          
      en: >
          Stops the threads and releases capture sources, models, and network sessions.
          
---
