---
uid: "9e400012"
id: neko-vrc-body.backend.perception.semantic.monitors
parent: neko-vrc-body.backend.perception.semantic
name: {zh: "显示器枚举", en: "Monitor Enumeration"}
description:
  zh: >
      枚举物理显示器虚拟桌面矩形，并找出包含采集区域中心的那一台。
      
  en: >
      Enumerates the virtual-desktop rectangles of physical monitors and finds the one containing the center of the capture region.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.718Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 1850
    end_line: 1910
apis:
  - protocol: file
    path: "backend/vision.py#_display_monitor_rects"
    description:
      zh: >
          枚举各物理显示器在虚拟桌面坐标系中的矩形。
          
      en: >
          Enumerates each physical monitor's rectangle in virtual-desktop coordinates.
          
  - protocol: file
    path: "backend/vision.py#_monitor_for_region"
    description:
      zh: >
          找出包含采集区域中心的那台显示器。
          
      en: >
          Finds the monitor that contains the center of the capture region.
          
---
