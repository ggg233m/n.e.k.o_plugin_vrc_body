---
uid: "9e400005"
id: neko-vrc-body.backend.perception.window-capture
parent: neko-vrc-body.backend.perception
name: {zh: "窗口捕获定位", en: "Window Capture Locating"}
description:
  zh: >
      定位目标窗口并判断它此刻是否真的可见，让采集只信任露在外面的画面。
      
  en: >
      Locates the target window and decides whether it is genuinely visible right now, so capture only trusts pixels that are actually on screen.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.724Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 330
    end_line: 569
---
