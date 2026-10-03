---
uid: "9e400006"
id: neko-vrc-body.backend.perception.window-capture.find
parent: neko-vrc-body.backend.perception.window-capture
name: {zh: "窗口查找", en: "Window Lookup"}
description:
  zh: >
      按标题查找 Windows 顶层窗口并判断它是否已最小化。
      
  en: >
      Finds a Windows top-level window by title and determines whether it has been minimized.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.723Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 330
    end_line: 430
apis:
  - protocol: file
    path: "backend/vision.py#find_window_region"
    description:
      zh: >
          按标题匹配窗口并返回其在虚拟桌面上的矩形区域。
          
      en: >
          Matches a window by title and returns its rectangle on the virtual desktop.
          
  - protocol: file
    path: "backend/vision.py#_window_minimized"
    description:
      zh: >
          判断该窗口句柄当前是否处于最小化状态。
          
      en: >
          Determines whether that window handle is currently minimized.
          
---
