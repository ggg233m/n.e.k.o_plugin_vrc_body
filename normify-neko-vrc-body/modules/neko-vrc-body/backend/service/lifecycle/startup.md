---
uid: 9e3e000b
id: neko-vrc-body.backend.service.lifecycle.startup
parent: neko-vrc-body.backend.service.lifecycle
name: {zh: "服务启动", en: "Service Start-up"}
description:
  zh: >
      服务启动：调度器、OSC、中继、导航器、动作时间轴与 VMC 校准流程。
      
  en: >
      Bringing the service up: scheduler, OSC, relays, navigator, action timeline and the VMC calibration pass.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.738Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 822
    end_line: 881
apis:
  - protocol: file
    path: "backend/service.py#BackendService.start"
    description:
      zh: >
          拉起服务持有的全部长期资源。
          
      en: >
          Bring up every long-lived resource the service owns.
          
  - protocol: file
    path: "backend/service.py#BackendService._navigator_complete_goal"
    description:
      zh: >
          处理导航器上报当前目标已完成。
          
      en: >
          Handle the navigator reporting that the active goal is complete.
          
---
