---
uid: "9e400007"
id: neko-vrc-body.backend.perception.window-capture.visibility
parent: neko-vrc-body.backend.perception.window-capture
name: {zh: "窗口可见性", en: "Window Visibility"}
description:
  zh: >
      报告目标窗口此刻是否真的露在外面，避免把被遮挡的桌面像素当成真实画面。
      
  en: >
      Reports whether the target window is actually exposed right now, avoiding treating occluded desktop pixels as the real picture.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.725Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 435
    end_line: 569
apis:
  - protocol: file
    path: "backend/vision.py#window_visibility"
    description:
      zh: >
          综合遮挡、最小化与屏幕边界，给出窗口此刻的可见性判定。
          
      en: >
          Combining occlusion, minimization, and screen bounds into the window's visibility verdict at this moment.
          
  - protocol: file
    path: "backend/vision.py#_normalize_region"
    description:
      zh: >
          把窗口矩形裁剪到屏幕范围内并保证宽高为正。
          
      en: >
          Clips a window rectangle to the screen bounds and guarantees positive width and height.
          
---
