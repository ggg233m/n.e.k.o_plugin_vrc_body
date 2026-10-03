---
uid: 9e3a03a1
id: neko-vrc-body.clips.expression-motion.phases
parent: neko-vrc-body.clips.expression-motion
name: {zh: "相位帧", en: "Phase Frames"}
description:
  zh: >
      预备与挥动相位帧，全部相对手势启动那一刻捕获的姿态。
      
  en: >
      The anticipation and stroke phase frames, relative to the pose captured when the gesture started.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.806Z"
fingerprint: c259af45ca5ae068f69fda4bf52ec86545024c5ec90b3354281011688df04f0f
source:
  - path: "expression_motion.py"
    line: 53
    end_line: 251
apis:
  - protocol: file
    path: "expression_motion.py#_phase_frames"
    description:
      zh: >
          相对手势起始姿态构造预备帧与挥动帧。
          
      en: >
          Build anticipation and stroke frames relative to the pose at gesture start.
          
  - protocol: file
    path: "expression_motion.py#sample_expression"
    description:
      zh: >
          按给定时间相对身体参数采样叠加层。
          
      en: >
          Sample the overlay at a given time against the body profile.
          
---
