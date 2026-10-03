---
uid: 9e3e001c
id: neko-vrc-body.backend.service.osc-api.turn-stop
parent: neko-vrc-body.backend.service.osc-api
name: {zh: "转向与停止", en: "Turn and Stop"}
description:
  zh: >
      通过 OSC 转向，以及保证不会留下任何移动状态的停止路径。
      
  en: >
      Turning over OSC and the stop-movement path that guarantees nothing is left moving.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.753Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2823
    end_line: 2879
apis:
  - protocol: file
    path: "backend/service.py#BackendService.set_turn"
    description:
      zh: >
          设置一段时间内的横向转向轴。
          
      en: >
          Set a horizontal turn axis for a duration.
          
  - protocol: file
    path: "backend/service.py#BackendService.stop_movement"
    description:
      zh: >
          立即停止所有移动。
          
      en: >
          Stop all movement immediately.
          
---
