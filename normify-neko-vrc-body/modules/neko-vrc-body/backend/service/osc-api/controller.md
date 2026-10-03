---
uid: 9e3e001d
id: neko-vrc-body.backend.service.osc-api.controller
parent: neko-vrc-body.backend.service.osc-api
name: {zh: "虚拟控制器", en: "Virtual Controllers"}
description:
  zh: >
      虚拟 AnyaDance Index 控制器：摇杆、按键，以及清空它们的释放路径。
      
  en: >
      The virtual AnyaDance Index controllers: stick, buttons and the release path that clears them.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.749Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2881
    end_line: 2977
apis:
  - protocol: file
    path: "backend/service.py#BackendService.set_controller_axes"
    description:
      zh: >
          设置单侧虚拟控制器摇杆。
          
      en: >
          Set the virtual controller stick on one side.
          
  - protocol: file
    path: "backend/service.py#BackendService.set_controller_button"
    description:
      zh: >
          带保持窗口设置虚拟控制器按键。
          
      en: >
          Set a virtual controller button with a hold window.
          
  - protocol: file
    path: "backend/service.py#BackendService.release_controller_inputs"
    description:
      zh: >
          释放单侧或全部仍按住的控制器输入。
          
      en: >
          Release held controller inputs on one side or all.
          
---
