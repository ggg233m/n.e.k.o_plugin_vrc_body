---
uid: "9e400003"
id: neko-vrc-body.backend.perception.overlay.geometry
parent: neko-vrc-body.backend.perception.overlay
name: {zh: "叠加几何换算", en: "Overlay Geometry"}
description:
  zh: >
      把归一化检测框换算成像素矩形，并丢掉画不出来的框。
      
  en: >
      Converts normalized detection boxes into pixel rectangles and drops the boxes that cannot be drawn.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.705Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 186
    end_line: 258
apis:
  - protocol: file
    path: "backend/vision.py#overlay_boxes_geometry"
    description:
      zh: >
          把归一化检测框换算为像素矩形列表，并剔除越界或退化项。
          
      en: >
          Converts normalized detection boxes into a list of pixel rectangles, dropping out-of-bounds or degenerate entries.
          
---
