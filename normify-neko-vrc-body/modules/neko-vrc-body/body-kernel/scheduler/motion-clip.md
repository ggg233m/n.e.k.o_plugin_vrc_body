---
uid: 9e3a032d
id: neko-vrc-body.body-kernel.scheduler.motion-clip
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "片段播放", en: "Clip Playback"}
description:
  zh: >
      从片段库播放 .nya 片段，含加载、基础姿态与完成记账。
      
  en: >
      Playing a .nya clip from the clip library, including load, base pose and completion bookkeeping.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.796Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 1282
    end_line: 1359
  - path: "scheduler.py"
    line: 1361
    end_line: 1399
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._start_clip"
    description:
      zh: >
          启动一次 .nya 片段播放。
          
      en: >
          Start a .nya clip playback.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._sample_active_motion"
    description:
      zh: >
          每一拍采样当前动作并推进其时间轴。
          
      en: >
          Sample the active motion each tick and advance its timeline.
          
deps:
  - kind: call
    to: neko-vrc-body.clips.nya.library
    label: {zh: "加载并采样片段", en: "Load and sample the clip"}
---
