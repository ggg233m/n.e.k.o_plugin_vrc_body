---
uid: 9e3e001f
id: neko-vrc-body.backend.service.clips
parent: neko-vrc-body.backend.service
name: {zh: "片段播放", en: "Clip Playback"}
description:
  zh: >
      列出动作片段，并通过片段库播放一次语义表情。
      
  en: >
      Listing motion clips and playing a semantic expression through the clip library.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.733Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 3059
    end_line: 3138
apis:
  - protocol: file
    path: "backend/service.py#BackendService.list_clips"
    description:
      zh: >
          列出可用的动作片段。
          
      en: >
          List the available motion clips.
          
  - protocol: file
    path: "backend/service.py#BackendService.semantic_express"
    description:
      zh: >
          通过片段库播放一次语义表情。
          
      en: >
          Play a semantic expression through the clip library.
          
---
