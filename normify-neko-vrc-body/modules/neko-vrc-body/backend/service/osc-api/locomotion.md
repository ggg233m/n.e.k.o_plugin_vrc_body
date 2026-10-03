---
uid: 9e3e001b
id: neko-vrc-body.backend.service.osc-api.locomotion
parent: neko-vrc-body.backend.service.osc-api
name: {zh: "移动", en: "Locomotion"}
description:
  zh: >
      带显式时长的移动轴，并记录命令是否真的发出去了。
      
  en: >
      Locomotion axes with an explicit duration, recording whether the command was actually sent.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.751Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2768
    end_line: 2821
apis:
  - protocol: file
    path: "backend/service.py#BackendService.set_locomotion"
    description:
      zh: >
          设置一段时间内的纵向与横向移动。
          
      en: >
          Set vertical and horizontal locomotion for a duration.
          
deps:
  - kind: call
    to: neko-vrc-body.device-io.osc.bridge-axes
    label: {zh: "设置摇杆轴", en: "Set joystick axes"}
---
