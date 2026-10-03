---
uid: 9e3a0355
id: neko-vrc-body.device-io.vmc.relay-frame
parent: neko-vrc-body.device-io.vmc
name: {zh: "帧提取", en: "Frame Extraction"}
description:
  zh: >
      把解算出的骨骼变换转成六点 AnyaDance 帧，以及中继的对外回读。
      
  en: >
      Converting solved bone transforms into a six-point AnyaDance frame, and the relay's public read-out.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.834Z"
fingerprint: a49ba23fb5ec0e9ebaa1ef04e8b0a3ded0b99c3a792d7b1568bce88cb271c578
source:
  - path: "vmc_idle.py"
    line: 537
    end_line: 717
apis:
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay._frame_from_world_locked"
    description:
      zh: >
          把解算出的世界变换转成 AnyaDance 帧。
          
      en: >
          Convert solved world transforms into an AnyaDance frame.
          
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay.latest_frame"
    description:
      zh: >
          读取最新解算出的身体帧。
          
      en: >
          Read the latest solved body frame.
          
  - protocol: file
    path: "vmc_idle.py#VmcIdleRelay.snapshot"
    description:
      zh: >
          读取中继快照，含基线与解算统计。
          
      en: >
          Read the relay snapshot, including baseline and solve statistics.
          
---
