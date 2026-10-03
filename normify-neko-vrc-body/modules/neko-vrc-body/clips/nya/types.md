---
uid: 9e3a0380
id: neko-vrc-body.clips.nya.types
parent: neko-vrc-body.clips.nya
name: {zh: "片段数据模型", en: "Clip Data Model"}
description:
  zh: >
      片段数据模型：一个关键帧、一个带循环模式与时长的片段，以及手指名表。
      
  en: >
      The clip data model: a keyframe, a clip with its loop mode and duration, and the finger name table.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.808Z"
fingerprint: af36750f4baf9f8ec574db149233d6e34f52a4cca5324558b7d87e202a6a0414
source:
  - path: "nya.py"
    line: 22
    end_line: 59
apis:
  - protocol: file
    path: "nya.py#NyaKeyframe"
    description:
      zh: >
          一个关键帧：时间戳加一整套设备变换。
          
      en: >
          One keyframe: timestamp plus a full set of device transforms.
          
  - protocol: file
    path: "nya.py#NyaClip"
    description:
      zh: >
          一个已解析的 .nya 片段：关键帧、循环模式与时长。
          
      en: >
          A parsed .nya clip: keyframes, loop mode and duration.
          
---
