---
uid: 9e3a0302
id: neko-vrc-body.body-kernel.model.frame
parent: neko-vrc-body.body-kernel.model
name: {zh: "身体帧", en: "Body Frame"}
description:
  zh: >
      身体帧本身、规范的中立 T-pose，以及输入清零。
      
  en: >
      The body frame itself, the canonical neutral T-pose and input neutralisation.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.784Z"
fingerprint: 172f824e6246028ebf80474b7ac6a938e3b15068a82f1e384a1b87744984bc0c
source:
  - path: "model.py"
    line: 86
    end_line: 110
apis:
  - protocol: file
    path: "model.py#FrameState"
    description:
      zh: >
          一帧完整身体姿态：全部设备加上派生关节。
          
      en: >
          One complete body pose: every device plus the derived joints.
          
  - protocol: file
    path: "model.py#neutral_frame"
    description:
      zh: >
          从 C++ 实现移植来的 AnyaDance 规范复位 T-pose。
          
      en: >
          AnyaDance's canonical reset T-pose, ported from the C++ implementation.
          
  - protocol: file
    path: "model.py#neutralize_inputs"
    description:
      zh: >
          把一帧里的控制器输入全部清零。
          
      en: >
          Zero every controller input in a frame.
          
---
