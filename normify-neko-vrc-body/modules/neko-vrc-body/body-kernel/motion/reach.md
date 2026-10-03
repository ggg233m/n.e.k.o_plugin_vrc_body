---
uid: 9e3a0313
id: neko-vrc-body.body-kernel.motion.reach
parent: neko-vrc-body.body-kernel.motion
name: {zh: "伸手目标", en: "Reach Target"}
description:
  zh: >
      构造伸手目标帧，含被裁切的检测框能被信任到多远。
      
  en: >
      Building the target frame for a reach, including how far a clipped detection box may honestly be trusted.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.788Z"
fingerprint: 87b9c47e51e1edf49546d81a698b1088f6f5798e561623d5a8cbfbb63a6e69be
source:
  - path: "motion.py"
    line: 327
    end_line: 358
apis:
  - protocol: file
    path: "motion.py#reach_target"
    description:
      zh: >
          朝某方向或某点构造伸手并合手的目标帧。
          
      en: >
          Build a reach-and-close target frame towards a direction or point.
          
---
