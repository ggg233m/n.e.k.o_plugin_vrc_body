---
uid: 9e3a0234
id: neko-vrc-body.plugin.tools.vision.frame
parent: neko-vrc-body.plugin.tools.vision
name: {zh: "视觉取帧", en: "Vision Frame"}
description:
  zh: >
      取最新帧作为图片部分，由拉帧预算限流，并可叠加检测框。
      
  en: >
      Pulling the latest frame as an image part, rate-limited by the frame budget and optionally overlaid with detections.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.897Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2847
    end_line: 2968
apis:
  - protocol: rpc
    path: "vrc_vision_frame"
    description:
      zh: >
          工具：取最新帧作为图片部分，可选叠加检测框。
          
      en: >
          Tool: fetch the latest frame as an image part, optionally with detection overlay.
          
  - protocol: file
    path: "__init__.py#_FRAME_MAX_BASE64_CHARS"
    description:
      zh: >
          服务该帧之前强制检查配置的最大帧龄。
          
      en: >
          Enforce the configured maximum frame age before serving.
          
---
