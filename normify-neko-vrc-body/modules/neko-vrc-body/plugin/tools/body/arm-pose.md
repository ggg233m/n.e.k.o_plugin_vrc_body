---
uid: 9e3a0225
id: neko-vrc-body.plugin.tools.body.arm-pose
parent: neko-vrc-body.plugin.tools.body
name: {zh: "预备姿态", en: "Arm Pose"}
description:
  zh: >
      预备一个命名的全身姿态，并通过调度器的目标姿态摘要解析。
      
  en: >
      Arming a named whole-body pose, resolving it through the scheduler's target-pose summary.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.883Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2256
    end_line: 2313
apis:
  - protocol: rpc
    path: "body_arm_pose"
    description:
      zh: >
          工具：以指定时长与缓动预备一个命名的全身姿态。
          
      en: >
          Tool: arm a named whole-body pose with a duration and easing.
          
---
