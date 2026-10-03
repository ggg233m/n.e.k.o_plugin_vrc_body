---
uid: "9e400015"
id: neko-vrc-body.backend.perception.semantic-worker
parent: neko-vrc-body.backend.perception
name: {zh: "语义工作线程", en: "Semantic Worker"}
description:
  zh: >
      VLM 单槽异步 worker，慢推理不阻塞检测与导航。
      
  en: >
      A single-slot asynchronous VLM worker whose slow inference does not block detection and navigation.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.714Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 2219
    end_line: 2328
apis:
  - protocol: file
    path: "backend/vision.py#SemanticWorker.submit"
    description:
      zh: >
          提交一个语义任务到单槽队列，忙时立即返回而不排队。
          
      en: >
          Submits a semantic job to the single-slot queue, returning immediately instead of queueing when busy.
          
  - protocol: file
    path: "backend/vision.py#SemanticWorker.status"
    description:
      zh: >
          报告队列占用、正在处理的任务与最近一次结果或错误。
          
      en: >
          Reports queue occupancy, the job in flight, and the most recent result or error.
          
---
