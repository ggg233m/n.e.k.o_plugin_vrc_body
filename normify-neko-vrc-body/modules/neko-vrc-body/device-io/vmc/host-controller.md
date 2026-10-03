---
uid: 9e3a0356
id: neko-vrc-body.device-io.vmc.host-controller
parent: neko-vrc-body.device-io.vmc
name: {zh: "宿主 VMC 控制器", en: "Host VMC Controller"}
description:
  zh: >
      驱动宿主的公开 VMC 输出 API：启用它、采集静止姿态基线，然后恢复它进入时的原状态。
      
  en: >
      Drives the host's documented VMC output API: enable it, capture a rest-pose baseline, then restore whatever state it found.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.831Z"
fingerprint: 3cd849a134a0dead77a616ea68edc36cbd05a40df41d7e667bba27370f4512a2
source:
  - path: "host_vmc.py"
    line: 19
    end_line: 251
apis:
  - protocol: file
    path: "host_vmc.py#HostVmcController.start"
    description:
      zh: >
          中继存活期间启用宿主 VMC 输出，并在结束后恢复原状。
          
      en: >
          Enable host VMC output while the relay is alive, then restore its prior state.
          
  - protocol: file
    path: "host_vmc.py#HostVmcController.stop"
    description:
      zh: >
          恢复控制器进入时发现的宿主 VMC 状态。
          
      en: >
          Restore the host VMC state the controller found on entry.
          
  - protocol: file
    path: "host_vmc.py#HostVmcController.calibrate_rest_pose"
    description:
      zh: >
          从宿主采集 T-pose 静止基线。
          
      en: >
          Capture a T-pose rest baseline from the host.
          
---
