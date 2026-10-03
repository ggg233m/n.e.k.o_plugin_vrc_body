---
uid: 9e3e001a
id: neko-vrc-body.backend.service.osc-api.parameter
parent: neko-vrc-body.backend.service.osc-api
name: {zh: "参数与脉冲", en: "Parameter and Pulse"}
description:
  zh: >
      化身参数写入，以及工具用于离散动作的短输入脉冲。
      
  en: >
      Avatar parameter writes and the short input pulses the tools use for discrete actions.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.752Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2702
    end_line: 2766
apis:
  - protocol: file
    path: "backend/service.py#BackendService.send_avatar_parameter"
    description:
      zh: >
          通过 OSC 设置化身表情参数。
          
      en: >
          Set an avatar expression parameter over OSC.
          
  - protocol: file
    path: "backend/service.py#BackendService.pulse_input"
    description:
      zh: >
          脉冲一次虚拟输入动作。
          
      en: >
          Pulse a virtual input action.
          
  - protocol: file
    path: "backend/service.py#BackendService.pulse_jump"
    description:
      zh: >
          脉冲一次跳跃动作。
          
      en: >
          Pulse the jump action.
          
---
