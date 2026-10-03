---
uid: 9e41001f
id: neko-vrc-body.backend.wgc-capture.session
parent: neko-vrc-body.backend.wgc-capture
name: {zh: "窗口捕获会话", en: "Window Capture Session"}
description:
  zh: >
      针对单个窗口句柄的捕获会话，构造成功即代表捕获已启动。
      
  en: >
      A capture session for a single window handle, where successful construction means capture has already started.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.772Z"
fingerprint: 0ff7a755ae59a0b73e3394a4115de54c92c1e6cd083b2f71b449a093a7cca46a
source:
  - path: "backend/wgc_capture.py"
    line: 123
    end_line: 360
apis:
  - protocol: file
    path: "backend/wgc_capture.py#WgcSession.read"
    description:
      zh: >
          取回当前窗口的最新捕获帧。
          
      en: >
          Retrieves the latest captured frame of the current window.
          
  - protocol: file
    path: "backend/wgc_capture.py#WgcSession._create_device"
    description:
      zh: >
          创建捕获与纹理拷贝共用的 D3D11 设备。
          
      en: >
          Creates the D3D11 device shared by capture and texture copy.
          
  - protocol: file
    path: "backend/wgc_capture.py#WgcSession._frame_to_array"
    description:
      zh: >
          把捕获帧映射并转成像素数组。
          
      en: >
          Maps a captured frame and converts it into a pixel array.
          
---
