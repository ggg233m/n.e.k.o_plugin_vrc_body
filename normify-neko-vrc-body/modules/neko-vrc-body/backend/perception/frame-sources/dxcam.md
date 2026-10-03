---
uid: 9e40000a
id: neko-vrc-body.backend.perception.frame-sources.dxcam
parent: neko-vrc-body.backend.perception.frame-sources
name: {zh: "DXcam 采集源", en: "DXcam Frame Source"}
description:
  zh: >
      DXcam 桌面镜像采集器，延迟导入并逐个尝试输出与捕获后端。
      
  en: >
      A DXcam desktop mirroring capture source, imported lazily and trying output and capture backends one by one.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.698Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 730
    end_line: 1050
apis:
  - protocol: file
    path: "backend/vision.py#DxcamFrameSource._build_candidates"
    description:
      zh: >
          枚举 DXcam 候选组合（输出设备与捕获方式）并排序。
          
      en: >
          Enumerates and orders DXcam candidate combinations of output device and capture method.
          
  - protocol: file
    path: "backend/vision.py#DxcamFrameSource._activate_candidate_locked"
    description:
      zh: >
          在锁保护下把某个候选切换为当前生效的 DXcam 相机。
          
      en: >
          Switches a candidate to the active DXcam camera under lock protection.
          
  - protocol: file
    path: "backend/vision.py#DxcamFrameSource.read"
    description:
      zh: >
          从当前 DXcam 相机读取一帧桌面镜像。
          
      en: >
          Reads one frame of the desktop mirror from the active DXcam camera.
          
---
