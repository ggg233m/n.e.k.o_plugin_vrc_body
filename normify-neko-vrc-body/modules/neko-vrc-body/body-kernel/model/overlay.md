---
uid: 9e3a0301
id: neko-vrc-body.body-kernel.model.overlay
parent: neko-vrc-body.body-kernel.model
name: {zh: "控制器输入叠加层", en: "Controller Input Overlay"}
description:
  zh: >
      叠在身体帧之上的后写入生效虚拟控制器输入层，本身不具权威性。
      
  en: >
      A latest-wins overlay of virtual controller input layered on top of a body frame, never authoritative.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.785Z"
fingerprint: 172f824e6246028ebf80474b7ac6a938e3b15068a82f1e384a1b87744984bc0c
source:
  - path: "model.py"
    line: 59
    end_line: 82
apis:
  - protocol: file
    path: "model.py#ControllerInputOverlay"
    description:
      zh: >
          后写入生效的虚拟控制器输入，叠加在身体帧之上。
          
      en: >
          Latest-wins virtual controller input layered over a body frame.
          
  - protocol: file
    path: "model.py#ControllerInputOverlay.clear"
    description:
      zh: >
          清空整个叠加层。
          
      en: >
          Clear the whole overlay.
          
  - protocol: file
    path: "model.py#ControllerInputOverlay.clear_side"
    description:
      zh: >
          只清空单侧的叠加层。
          
      en: >
          Clear the overlay for one side only.
          
---
