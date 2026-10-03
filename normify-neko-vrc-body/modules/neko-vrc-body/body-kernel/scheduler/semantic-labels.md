---
uid: 9e3a0331
id: neko-vrc-body.body-kernel.scheduler.semantic-labels
parent: neko-vrc-body.body-kernel.scheduler
name: {zh: "语义姿态标签", en: "Semantic Pose Labels"}
description:
  zh: >
      把原始四元数转成模型能推理的词：方向、仰角与手部姿态标签。
      
  en: >
      Turning raw quaternions into words a model can reason about: direction, elevation and hand-pose labels.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.800Z"
fingerprint: e40a02bbd6368d5bac890a2a56301f572e16d08334eb2dd09f6a7325121f9820
source:
  - path: "scheduler.py"
    line: 1564
    end_line: 1656
apis:
  - protocol: file
    path: "scheduler.py#BodyScheduler._semantic_pose"
    description:
      zh: >
          用语义词而不是数字渲染当前姿态。
          
      en: >
          Render the current pose in semantic words rather than numbers.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._quat_to_euler_deg"
    description:
      zh: >
          把四元数转成 yaw/pitch/roll 角度。
          
      en: >
          Convert a quaternion to yaw/pitch/roll degrees.
          
  - protocol: file
    path: "scheduler.py#BodyScheduler._direction_label"
    description:
      zh: >
          给某个方向起一个方位角分档名。
          
      en: >
          Name the azimuth bucket a direction falls into.
          
---
