---
uid: 9e3e000c
id: neko-vrc-body.backend.service.lifecycle.vmc-calibration
parent: neko-vrc-body.backend.service.lifecycle
name: {zh: "VMC 校准", en: "VMC Calibration"}
description:
  zh: >
      何时以及如何采集 VMC 静止姿态基线，含长动作之后开启的窗口。
      
  en: >
      When and how the VMC rest-pose baseline is captured, including the window opened after a long motion.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.739Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 883
    end_line: 1032
apis:
  - protocol: file
    path: "backend/service.py#BackendService._start_vmc_calibration"
    description:
      zh: >
          启动 VMC 静止姿态校准流程。
          
      en: >
          Start the VMC rest-pose calibration pass.
          
  - protocol: file
    path: "backend/service.py#BackendService.vmc_recalibrate"
    description:
      zh: >
          重新执行校准，可选择接受当前姿态。
          
      en: >
          Re-run calibration, optionally accepting the current pose.
          
  - protocol: file
    path: "backend/service.py#BackendService._stop_vmc_calibration"
    description:
      zh: >
          停止校准流程。
          
      en: >
          Stop the calibration pass.
          
  - protocol: file
    path: "backend/service.py#BackendService._on_motion_started"
    description:
      zh: >
          在长动作开始时开启一次重新校准窗口。
          
      en: >
          Open a recalibration window when a long motion starts.
          
---
