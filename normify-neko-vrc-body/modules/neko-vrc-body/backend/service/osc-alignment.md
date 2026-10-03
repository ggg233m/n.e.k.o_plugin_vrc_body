---
uid: 9e3e0004
id: neko-vrc-body.backend.service.osc-alignment
parent: neko-vrc-body.backend.service
name: {zh: "OSC 对齐与驱动阻断", en: "OSC Alignment and Drive Block"}
description:
  zh: >
      让 OSC 速度样本与视觉帧时刻对齐，以及在信号不可信时阻断 navmesh 移动的规则。
      
  en: >
      Keeping OSC velocity samples aligned to visual frame timestamps, and the drive-block rule that stops navmesh locomotion when the signal is not trustworthy.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.748Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 482
    end_line: 543
apis:
  - protocol: file
    path: "backend/service.py#BackendService._sync_osc_alignment_samples"
    description:
      zh: >
          把近期 OSC 速度样本推进对齐缓冲。
          
      en: >
          Push recent OSC velocity samples into the alignment buffer.
          
  - protocol: file
    path: "backend/service.py#BackendService._vision_motion_feedback"
    description:
      zh: >
          读取供感知 worker 消费的动作反馈。
          
      en: >
          Read the motion feedback the perception worker consumes.
          
  - protocol: file
    path: "backend/service.py#BackendService._navmesh_motion_history"
    description:
      zh: >
          读取在线导航器所积分的近期速度历史。
          
      en: >
          Read the recent velocity history the online navigator integrates.
          
  - protocol: file
    path: "backend/service.py#BackendService._navmesh_drive_block"
    description:
      zh: >
          在必须阻断 navmesh 移动时返回原因字符串。
          
      en: >
          Return a reason string when navmesh locomotion must be blocked.
          
---
