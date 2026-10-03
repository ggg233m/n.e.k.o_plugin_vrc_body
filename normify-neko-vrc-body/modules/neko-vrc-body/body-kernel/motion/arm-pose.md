---
uid: 9e3a0312
id: neko-vrc-body.body-kernel.motion.arm-pose
parent: neko-vrc-body.body-kernel.motion
name: {zh: "臂部与手部姿态", en: "Arm and Hand Poses"}
description:
  zh: >
      臂部姿态目标、相对手部移动，以及把命名手部姿态施加到帧上。
      
  en: >
      Arm pose targets, relative hand movement and applying a named hand pose to a frame.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.786Z"
fingerprint: 87b9c47e51e1edf49546d81a698b1088f6f5798e561623d5a8cbfbb63a6e69be
source:
  - path: "motion.py"
    line: 202
    end_line: 324
apis:
  - protocol: file
    path: "motion.py#arm_pose_target"
    description:
      zh: >
          某个命名姿态对应的目标臂部姿态。
          
      en: >
          Target arm pose for a named posture.
          
  - protocol: file
    path: "motion.py#move_hand_target"
    description:
      zh: >
          把控制器移动到相对稳定身体锚点的位置。
          
      en: >
          Move a controller to a position relative to a stable body anchor.
          
  - protocol: file
    path: "motion.py#apply_hand_pose"
    description:
      zh: >
          按力度把一个手部姿态施加到帧上。
          
      en: >
          Apply a hand pose to a frame, scaled by strength.
          
---
