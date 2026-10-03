---
uid: 9e3d020a
id: neko-vrc-body.backend.nav-loop.config
parent: neko-vrc-body.backend.nav-loop
name: {zh: "回环配置与关键帧特征", en: "Loop Config"}
description:
  zh: >
      回环检测配置与关键帧特征容器；只优化平移，朝向来自 HMD。
      
  en: >
      Loop closure detection configuration and keyframe feature container; only translation is optimized, orientation comes from the HMD.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.624Z"
fingerprint: 0d562774afcbca2772a8abb5b0d193fdfb369fdf0074750d5fb80d71774492c7
source:
  - path: "backend/nav_loop.py"
    line: 36
    end_line: 89
apis:
  - protocol: file
    path: "backend/nav_loop.py#LoopConfig"
    description:
      zh: >
          回环检测配置，含优化方法选项与阈值。
          
      en: >
          Loop closure detection configuration, including optimizer choice and thresholds.
          
  - protocol: file
    path: "backend/nav_loop.py#KeyframeFeatures"
    description:
      zh: >
          关键帧特征容器，保存位姿、描述子与匹配项。
          
      en: >
          Keyframe feature container holding pose, descriptors, and matches.
          
---
