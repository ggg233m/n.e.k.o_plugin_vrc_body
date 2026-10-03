---
uid: 9e3a0322
id: neko-vrc-body.body-kernel.scheduler.lifecycle
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "线程生命周期", en: "Thread Lifecycle"}
description:
  zh: >
      调度线程的构造、初始快照，以及启动/停止/存活/快照这几个对外方法。
      
  en: >
      Construction, the initial snapshot and the start/stop/alive/snapshot surface of the scheduler thread.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.795Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 153
    end_line: 324
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler.start"
    description:
      zh: >
          启动实时调度线程。
          
      en: >
          Start the real-time scheduler thread.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler.shutdown"
    description:
      zh: >
          停止调度线程并释放传输层。
          
      en: >
          Stop the scheduler thread and release the transport.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler.thread_alive"
    description:
      zh: >
          调度线程是否仍然存活。
          
      en: >
          Whether the scheduler thread is still alive.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler.snapshot"
    description:
      zh: >
          读取调度器的公开快照。
          
      en: >
          Read the scheduler's public snapshot.
          
---
