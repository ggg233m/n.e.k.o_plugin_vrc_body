---
uid: 9e3a0362
id: neko-vrc-body.device-io.driver-log.timebase
parent: neko-vrc-body.device-io.driver-log
name: {zh: "视频时间基准", en: "Video Timebase"}
description:
  zh: >
      共享的视频时间基准，把帧索引与动作事件锁在同一单调时钟上。
      
  en: >
      The shared video timebase that locks frame indices and action events onto one monotonic clock.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.821Z"
fingerprint: c8a4fd5ddeaad285f69e1b67a07314be7c0738f57a94deaa19af7b24210179e9
source:
  - path: "driver_log.py"
    line: 567
    end_line: 616
apis:
  - protocol: file
    path: "driver_log.py#VideoTimebase"
    description:
      zh: >
          把视频帧索引与动作事件锁在同一个单调时钟基准上。
          
      en: >
          Lock a video frame index and action events onto one monotonic clock base.
          
  - protocol: file
    path: "driver_log.py#VideoTimebase.frame_timestamp"
    description:
      zh: >
          某个视频帧索引对应的单调时间戳。
          
      en: >
          Monotonic timestamp of a video frame index.
          
  - protocol: file
    path: "driver_log.py#VideoTimebase.frame_index_at"
    description:
      zh: >
          给定单调时间对应的视频帧索引。
          
      en: >
          Video frame index at a given monotonic time.
          
---
