---
uid: "9e420016"
id: neko-vrc-body.backend.nav-online.session-recorder
parent: neko-vrc-body.backend.nav-online
name: {zh: "会话录制器", en: "Session recorder"}
description:
  zh: >
      默认关闭的原始输入落盘，让长时间场景能按同样顺序离线重放。
      
  en: >
      Optional raw input dumping that is off by default so long scenarios can be replayed offline in the same order.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.654Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 166
    end_line: 270
apis:
  - protocol: file
    path: "backend/nav_online.py#SessionRecorder.keyframe"
    description:
      zh: >
          把一个关键帧负载追加进录制文件。
          
      en: >
          Appends one keyframe payload to the recording.
          
  - protocol: file
    path: "backend/nav_online.py#SessionRecorder.close"
    description:
      zh: >
          关闭录制并把数据落盘。
          
      en: >
          Closes the recording and flushes it to disk.
          
  - protocol: file
    path: "backend/nav_online.py#SessionRecorder.status"
    description:
      zh: >
          报告录制器状态与已写入的负载数量。
          
      en: >
          Reports the recorder state and its written payload count.
          
---
