---
uid: 9e41001c
id: neko-vrc-body.backend.openvr-mirror.stereo
parent: neko-vrc-body.backend.openvr-mirror
name: {zh: "立体画面读取", en: "Stereo Frame Read"}
description:
  zh: >
      取同一合成帧的双眼画面，并从投影矩阵反推内参。
      
  en: >
      Takes both eyes from the same composited frame and back-derives the intrinsics from the projection matrices.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.692Z"
fingerprint: f3d92ecec9db56dafe3d7aa0aaa351a8790ddcceb20a157ad8745cebf1318124
source:
  - path: "backend/openvr_mirror.py"
    line: 255
    end_line: 299
apis:
  - protocol: file
    path: "backend/openvr_mirror.py#read_stereo"
    description:
      zh: >
          从同一个合成帧中读出左右眼画面。
          
      en: >
          Reads the left and right eye images out of one composited frame.
          
  - protocol: file
    path: "backend/openvr_mirror.py#projection_intrinsics"
    description:
      zh: >
          由 OpenVR 投影矩阵反推相机内参。
          
      en: >
          Back-derives the camera intrinsics from the OpenVR projection matrices.
          
  - protocol: file
    path: "backend/openvr_mirror.py#create_device"
    description:
      zh: >
          创建用于纹理回读的 D3D11 设备。
          
      en: >
          Creates the D3D11 device used for texture readback.
          
---
