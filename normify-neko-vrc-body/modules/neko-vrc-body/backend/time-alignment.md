---
uid: 9e3a0044
id: neko-vrc-body.backend.time-alignment
parent: neko-vrc-body.backend
name: {zh: "时间对齐", en: "Time Alignment"}
description:
  zh: >
      把 OSC 速度样本对齐到视觉帧时刻，两者共用同一个单调时钟域。
      
  en: >
      Aligns OSC velocity samples to the timestamp of a captured visual frame, on one monotonic clock domain.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.763Z"
fingerprint: 6b7cc0a942de88140a5a00d26510840682a6b13d04bb4185f7f9016101d0df5e
source:
  - path: "backend/time_alignment.py"
    line: 14
    end_line: 120
apis:
  - protocol: file
    path: "backend/time_alignment.py#TimeAlignmentBuffer.sample_motion"
    description:
      zh: >
          取某个视觉帧时刻生效的速度样本。
          
      en: >
          Sample the velocity that applied at a given visual frame timestamp.
          
  - protocol: file
    path: "backend/time_alignment.py#TimeAlignmentBuffer.add_motion"
    description:
      zh: >
          把带时间戳的 OSC 速度样本加入环形缓冲。
          
      en: >
          Add a timestamped OSC velocity sample to the ring buffer.
          
---
