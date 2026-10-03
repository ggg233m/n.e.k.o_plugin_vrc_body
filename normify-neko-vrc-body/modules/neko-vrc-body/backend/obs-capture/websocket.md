---
uid: "9e410019"
id: neko-vrc-body.backend.obs-capture.websocket
parent: neko-vrc-body.backend.obs-capture
name: {zh: "OBS WebSocket 采集", en: "OBS WebSocket Capture"}
description:
  zh: >
      通过 obs-websocket v5 截图的兼容采集路径，刻意做成拉取式。
      
  en: >
      A compatibility capture path that pulls screenshots through obs-websocket v5, deliberately pull-based.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.690Z"
fingerprint: a425bb8c95520acbcda535ab5dc98f10420a4e78c156c4b6a9e006aceea078f9
source:
  - path: "backend/obs_capture.py"
    line: 353
    end_line: 593
apis:
  - protocol: file
    path: "backend/obs_capture.py#ObsWebSocketFrameSource._request_screenshot"
    description:
      zh: >
          向 obs-websocket 请求一张截图并解码为帧。
          
      en: >
          Requests a screenshot from obs-websocket and decodes it into a frame.
          
  - protocol: file
    path: "backend/obs_capture.py#ObsWebSocketFrameSource._connect"
    description:
      zh: >
          建立并完成 obs-websocket v5 的鉴权握手。
          
      en: >
          Establishes the obs-websocket v5 connection and completes its authentication handshake.
          
  - protocol: file
    path: "backend/obs_capture.py#ObsWebSocketFrameSource.status"
    description:
      zh: >
          上报 WebSocket 采集通道的连接与取帧状态。
          
      en: >
          Reports the connection and frame-fetch status of the WebSocket capture channel.
          
---
