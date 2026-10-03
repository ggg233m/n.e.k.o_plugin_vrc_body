---
uid: 9e40001d
id: neko-vrc-body.backend.perception.runtime.frame-cache
parent: neko-vrc-body.backend.perception.runtime
name: {zh: "帧缓存与叠加", en: "Frame Cache and Overlay"}
description:
  zh: >
      帧缓存与叠加渲染，带显式的最大帧龄约束。
      
  en: >
      Frame caching and overlay rendering with an explicit maximum frame-age bound.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.707Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 3408
    end_line: 3698
apis:
  - protocol: file
    path: "backend/vision.py#VisionRuntime.latest_frame"
    description:
      zh: >
          读取最新缓存帧，超过最大帧龄时返回空。
          
      en: >
          Reads the newest cached frame, returning nothing once the maximum frame age is exceeded.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime._apply_overlay"
    description:
      zh: >
          在缓存帧上按需叠加检测框并重新编码。
          
      en: >
          Overlays detection boxes onto the cached frame on demand and re-encodes it.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime._cache_frame"
    description:
      zh: >
          把新帧写入缓存，淘汰超过最大帧龄的旧帧。
          
      en: >
          Writes a new frame into the cache, evicting older frames beyond the maximum frame age.
          
---
