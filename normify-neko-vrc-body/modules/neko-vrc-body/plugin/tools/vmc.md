---
uid: 9e3a0239
id: neko-vrc-body.plugin.tools.vmc
parent: neko-vrc-body.plugin.tools
name: {zh: "VMC 重新校准", en: "VMC Recalibration"}
description:
  zh: >
      重新执行 VMC 静止姿态校准，可选择把当前姿态作为新基线。
      
  en: >
      Re-running the VMC rest-pose calibration, optionally accepting the current pose as the new baseline.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.900Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2631
    end_line: 2661
apis:
  - protocol: rpc
    path: "vmc_recalibrate"
    description:
      zh: >
          工具：重新执行 VMC 静止姿态校准。
          
      en: >
          Tool: re-run the VMC rest-pose calibration.
          
---
