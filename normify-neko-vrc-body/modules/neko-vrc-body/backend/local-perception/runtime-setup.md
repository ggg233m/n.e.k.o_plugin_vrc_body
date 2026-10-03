---
uid: "9e410000"
id: neko-vrc-body.backend.local-perception.runtime-setup
parent: neko-vrc-body.backend.local-perception
name: {zh: "推理运行时预热", en: "Inference Runtime Warmup"}
description:
  zh: >
      在采集栈之前拉起推理运行时，并收住进程级 OpenMP 线程池。
      
  en: >
      Brings the inference runtime up ahead of the capture stack and clamps the process-wide OpenMP thread pool.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.615Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 53
    end_line: 195
apis:
  - protocol: file
    path: "backend/local_perception.py#preload_inference_runtime"
    description:
      zh: >
          在采集流程开始前预加载并初始化推理运行时。
          
      en: >
          Preloads and initializes the inference runtime before the capture pipeline starts.
          
  - protocol: file
    path: "backend/local_perception.py#cap_openmp_threads"
    description:
      zh: >
          限制 OpenMP 线程池规模，避免与主线程争抢时间片。
          
      en: >
          Clamps the OpenMP thread pool so it does not starve the main thread.
          
  - protocol: file
    path: "backend/local_perception.py#openmp_thread_count"
    description:
      zh: >
          依据可用 CPU 数量推算安全的 OpenMP 线程上限。
          
      en: >
          Derives a safe OpenMP thread ceiling from the available CPU count.
          
---
