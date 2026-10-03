---
uid: 9e3e0014
id: neko-vrc-body.backend.service.submit
parent: neko-vrc-body.backend.service
name: {zh: "命令提交与回读", en: "Command Submit and Read-outs"}
description:
  zh: >
      身体命令入口与各回读：快照、世界增量、感知、aware 与控制延迟指标。
      
  en: >
      The body command entry point plus the read-outs: snapshot, world delta, perception, awareness and control latency metrics.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.756Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2145
    end_line: 2377
apis:
  - protocol: file
    path: "backend/service.py#BackendService.submit"
    description:
      zh: >
          提交身体命令并返回完成句柄。
          
      en: >
          Submit a body command and return a completion future handle.
          
  - protocol: file
    path: "backend/service.py#BackendService.snapshot"
    description:
      zh: >
          读取完整服务快照。
          
      en: >
          Read the full service snapshot.
          
  - protocol: file
    path: "backend/service.py#BackendService.world_delta"
    description:
      zh: >
          读取自某修订以来的世界增量。
          
      en: >
          Read the world delta since a revision.
          
  - protocol: file
    path: "backend/service.py#BackendService.perception"
    description:
      zh: >
          读取当前感知状态。
          
      en: >
          Read the current perception status.
          
  - protocol: file
    path: "backend/service.py#BackendService.awareness"
    description:
      zh: >
          读取化身 aware 状态。
          
      en: >
          Read the avatar awareness state.
          
  - protocol: file
    path: "backend/service.py#BackendService.record_control_dispatch"
    description:
      zh: >
          为控制下发计时并回报其延迟。
          
      en: >
          Time a control dispatch and report its latency.
          
deps:
  - kind: call
    to: neko-vrc-body.body-kernel.scheduler.submit
    label: {zh: "在调度器入队", en: "Enqueue on scheduler"}
  - kind: call
    to: neko-vrc-body.backend.world-state.store.query
    label: {zh: "读取世界增量", en: "Read world delta"}
---
