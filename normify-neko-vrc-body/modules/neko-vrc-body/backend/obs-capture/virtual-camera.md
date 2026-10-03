---
uid: "9e410018"
id: neko-vrc-body.backend.obs-capture.virtual-camera
parent: neko-vrc-body.backend.obs-capture
name: {zh: "OBS 虚拟摄像头采集", en: "OBS Virtual Camera Capture"}
description:
  zh: >
      以最新帧交接方式读取 OBS 虚拟摄像头，OpenCV 惰性导入。
      
  en: >
      Reads the OBS virtual camera with latest-frame handoff and a lazy OpenCV import.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.689Z"
fingerprint: a425bb8c95520acbcda535ab5dc98f10420a4e78c156c4b6a9e006aceea078f9
source:
  - path: "backend/obs_capture.py"
    line: 45
    end_line: 350
apis:
  - protocol: file
    path: "backend/obs_capture.py#ObsVirtualCameraFrameSource.read"
    description:
      zh: >
          读取最新可用帧并交给调用方，避免积压旧帧。
          
      en: >
          Reads the newest available frame for the caller instead of queueing stale ones.
          
  - protocol: file
    path: "backend/obs_capture.py#ObsVirtualCameraFrameSource._reconnect"
    description:
      zh: >
          设备断开或 OBS 重启后重建摄像头句柄。
          
      en: >
          Rebuilds the camera handle after a disconnect or an OBS restart.
          
  - protocol: file
    path: "backend/obs_capture.py#ObsVirtualCameraFrameSource.status"
    description:
      zh: >
          上报虚拟摄像头当前的连接与帧状态。
          
      en: >
          Reports the current connection and frame status of the virtual camera.
          
---
