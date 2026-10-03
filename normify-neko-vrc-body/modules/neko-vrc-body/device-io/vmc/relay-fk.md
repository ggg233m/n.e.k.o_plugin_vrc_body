---
uid: 9e3a0354
id: neko-vrc-body.device-io.vmc.relay-fk
parent: neko-vrc-body.device-io.vmc
name: {zh: "人形正向运动学", en: "Humanoid Forward Kinematics"}
description:
  zh: >
      人形正向运动学：解析世界变换，并判定一帧何时足够完整可用。
      
  en: >
      Humanoid forward kinematics: resolve the world transforms and decide when a frame is complete enough to use.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.833Z"
fingerprint: a49ba23fb5ec0e9ebaa1ef04e8b0a3ded0b99c3a792d7b1568bce88cb271c578
source:
  - path: "vmc_idle.py"
    line: 428
    end_line: 535
apis:
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay._world_transforms_locked"
    description:
      zh: >
          把全部骨骼解析成世界变换。
          
      en: >
          Resolve every bone into its world transform.
          
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay._finalize_pending_locked"
    description:
      zh: >
          在必需骨骼到齐后完成待定骨架。
          
      en: >
          Finalise a pending skeleton once the required bones arrive.
          
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay._is_t_pose_candidate_locked"
    description:
      zh: >
          判定当前姿态能否作为 T-pose 基线。
          
      en: >
          Decide whether the current pose can serve as the T-pose baseline.
          
---
