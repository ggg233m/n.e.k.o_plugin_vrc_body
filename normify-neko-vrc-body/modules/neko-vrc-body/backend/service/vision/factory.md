---
uid: 9e3e0007
id: neko-vrc-body.backend.service.vision.factory
parent: neko-vrc-body.backend.service.vision
name: {zh: "视觉 Worker 工厂", en: "Vision Worker Factory"}
description:
  zh: >
      创建、探测与退役感知 worker，含诚实的「未配置」回报。
      
  en: >
      Creating, probing and retiring the perception worker, including the honest "not configured" report.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.760Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 622
    end_line: 748
apis:
  - protocol: file
    path: "backend/service.py#BackendService._new_vision_worker"
    description:
      zh: >
          由画面源与检测器构建视觉 worker。
          
      en: >
          Build a vision worker from a frame source and detector.
          
  - protocol: file
    path: "backend/service.py#BackendService._build_configured_vision_source"
    description:
      zh: >
          按配置构建指定的画面源。
          
      en: >
          Build the frame source named by the configuration.
          
  - protocol: file
    path: "backend/service.py#BackendService._fresh_vision_source"
    description:
      zh: >
          构建一个刚探测过的画面源。
          
      en: >
          Build a freshly probed frame source.
          
  - protocol: file
    path: "backend/service.py#BackendService._vision_worker_not_configured"
    description:
      zh: >
          回报感知尚未配置。
          
      en: >
          Report that perception is not configured.
          
  - protocol: file
    path: "backend/service.py#BackendService._vision_worker_status"
    description:
      zh: >
          读取视觉 worker 状态。
          
      en: >
          Read the vision worker status.
          
  - protocol: file
    path: "backend/service.py#BackendService._stop_vision_worker_locked"
    description:
      zh: >
          停止视觉 worker 并释放其画面源。
          
      en: >
          Stop the vision worker and release its frame source.
          
---
