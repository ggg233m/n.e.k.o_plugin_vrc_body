---
uid: 9e3c010b
id: neko-vrc-body.backend.client.remote-scheduler
parent: neko-vrc-body.backend.client
name: {zh: "调度器代理", en: "Scheduler Proxy"}
description:
  zh: >
      调度器的 IPC 代理：提交身体命令并读取快照。
      
  en: >
      The scheduler's IPC proxy: submits body commands and reads snapshots.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.584Z"
fingerprint: 8f6d5a4ca8fb37bed50fe66446dfcb2eb54d67a9caca0125f881ac24d9844a7e
source:
  - path: "backend/client.py"
    line: 397
    end_line: 461
apis:
  - protocol: file
    path: "backend/client.py#RemoteScheduler.submit"
    description:
      zh: >
          向远端调度器提交一条身体命令。
          
      en: >
          Submits a body command to the remote scheduler.
          
  - protocol: file
    path: "backend/client.py#RemoteScheduler.snapshot"
    description:
      zh: >
          读取远端调度器状态快照。
          
      en: >
          Reads a snapshot of the remote scheduler state.
          
---
