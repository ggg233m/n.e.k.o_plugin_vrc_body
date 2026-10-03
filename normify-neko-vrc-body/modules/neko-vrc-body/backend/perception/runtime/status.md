---
uid: 9e40001f
id: neko-vrc-body.backend.perception.runtime.status
parent: neko-vrc-body.backend.perception.runtime
name: {zh: "后端状态汇总", en: "Backend Status Aggregation"}
description:
  zh: >
      汇总各后端状态，并允许运行时更换检测器与语义后端。
      
  en: >
      Aggregates the status of every backend and allows the runtime to swap its detector and semantic backend.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.712Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 3760
    end_line: 3904
apis:
  - protocol: file
    path: "backend/vision.py#VisionRuntime.status"
    description:
      zh: >
          汇总采集源、检测器、语义后端与线程的运行状态。
          
      en: >
          Aggregates the runtime status of capture sources, detector, semantic backend, and threads.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime.set_backends"
    description:
      zh: >
          热更换检测器与语义后端而不重启整个运行时。
          
      en: >
          Hot-swaps the detector and semantic backend without restarting the whole runtime.
          
---
