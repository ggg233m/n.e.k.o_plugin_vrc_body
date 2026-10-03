---
uid: 9e3a0352
id: neko-vrc-body.device-io.vmc.relay-calibration
parent: neko-vrc-body.device-io.vmc
name: {zh: "静止姿态校准", en: "Rest Pose Calibration"}
description:
  zh: >
      静止姿态基线处理：保持、重置，以及 T-pose 接受窗口。
      
  en: >
      Rest-pose baseline handling: hold, reset, and the T-pose acceptance window.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.833Z"
fingerprint: a49ba23fb5ec0e9ebaa1ef04e8b0a3ded0b99c3a792d7b1568bce88cb271c578
source:
  - path: "vmc_idle.py"
    line: 231
    end_line: 293
apis:
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay.hold_calibration"
    description:
      zh: >
          保持校准等待，直到出现静止姿态。
          
      en: >
          Hold calibration until a rest pose is available.
          
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay.reset_calibration"
    description:
      zh: >
          带原因重置校准状态。
          
      en: >
          Reset the calibration state with a reason.
          
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay.needs_recalibration"
    description:
      zh: >
          中继当前是否需要重新校准。
          
      en: >
          Whether the relay currently needs to recalibrate.
          
---
