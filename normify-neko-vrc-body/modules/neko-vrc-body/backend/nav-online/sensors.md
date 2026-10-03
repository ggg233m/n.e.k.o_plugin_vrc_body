---
uid: "9e420018"
id: neko-vrc-body.backend.nav-online.sensors
parent: neko-vrc-body.backend.nav-online
name: {zh: "OpenVR 传感器", en: "OpenVR sensors"}
description:
  zh: >
      同一 OpenVR 会话里取 HMD 姿态与镜像双目，并统计正前方近处障碍点数。
      
  en: >
      Reads the HMD pose and mirrored stereo pair from one OpenVR session and counts nearby obstacle points straight ahead.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.654Z"
fingerprint: 6f70a4f626aa87ec6441c8480e95442832cd6a82c4d8c2082201634766424359
source:
  - path: "backend/nav_online.py"
    line: 404
    end_line: 507
apis:
  - protocol: file
    path: "backend/nav_online.py#OpenVRSensors.open"
    description:
      zh: >
          为传感器打开一个共享的 OpenVR 会话。
          
      en: >
          Opens a shared OpenVR session for the sensors.
          
  - protocol: file
    path: "backend/nav_online.py#OpenVRSensors.read_stereo_rgb"
    description:
      zh: >
          从同一会话读取镜像双目 RGB 图。
          
      en: >
          Reads the mirrored stereo RGB pair from the same session.
          
  - protocol: file
    path: "backend/nav_online.py#near_obstacle"
    description:
      zh: >
          统计正前方近处区域中的障碍点数。
          
      en: >
          Counts obstacle points in the near area straight ahead.
          
---
