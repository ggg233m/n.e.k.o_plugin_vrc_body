---
uid: "9e400004"
id: neko-vrc-body.backend.perception.overlay.drawing
parent: neko-vrc-body.backend.perception.overlay
name: {zh: "检测框绘制", en: "Detection Box Drawing"}
description:
  zh: >
      在 JPEG 副本上绘制检测框，并回报实际画出的框数。
      
  en: >
      Draws detection boxes onto a copy of the JPEG and reports how many boxes were actually drawn.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.704Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 261
    end_line: 327
apis:
  - protocol: file
    path: "backend/vision.py#draw_detection_overlay"
    description:
      zh: >
          在帧的副本上绘制检测框与标签，返回编码后的新图与实际绘制数量。
          
      en: >
          Draws detection boxes and labels on a copy of the frame, returning the newly encoded image and the number actually drawn.
          
---
