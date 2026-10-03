---
uid: "9e430001"
id: neko-vrc-body.backend.nav-memory.session-writer
parent: neko-vrc-body.backend.nav-memory
name: {zh: "会话记忆写入器", en: "Session Memory Writer"}
description:
  zh: >
      一次导航会话的记忆写入器：所有公开方法非阻塞、不抛异常，出错只记进状态。
      
  en: >
      The memory writer of one navigation session: every public method is non-blocking and never raises, recording errors only in its status.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.639Z"
fingerprint: 8ac3ac2937f0825e7d07c1267ffa2e59000b81e0743552690708cb1b776bbb0e
source:
  - path: "backend/nav_memory.py"
    line: 118
    end_line: 277
apis:
  - protocol: file
    path: "backend/nav_memory.py#SessionWriter.keyframe"
    description:
      zh: >
          追加一个关键帧（位姿、图像与特征）。
          
      en: >
          Appends one keyframe carrying pose, image and features.
          
  - protocol: file
    path: "backend/nav_memory.py#SessionWriter.close"
    description:
      zh: >
          封口会话并把缓冲区落盘。
          
      en: >
          Seals the session and flushes its buffers to disk.
          
  - protocol: file
    path: "backend/nav_memory.py#SessionWriter.poses"
    description:
      zh: >
          返回该会话已记录的关键帧位姿。
          
      en: >
          Returns the keyframe poses recorded for the session.
          
  - protocol: file
    path: "backend/nav_memory.py#SessionWriter.status"
    description:
      zh: >
          返回写入器状态，含最后一条错误。
          
      en: >
          Returns the writer status, including the last recorded error.
          
---
