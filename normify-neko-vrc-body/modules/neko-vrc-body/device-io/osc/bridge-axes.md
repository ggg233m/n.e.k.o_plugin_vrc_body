---
uid: 9e3a0346
id: neko-vrc-body.device-io.osc.bridge-axes
parent: neko-vrc-body.device-io.osc
name: {zh: "摇杆轴", en: "Joystick Axes"}
description:
  zh: >
      带自动过期的摇杆轴控制，使导航决策永远不会把某个轴卡在打开状态。
      
  en: >
      Joystick axis control with auto-expiry, so a navigation decision can never leave an axis stuck on.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.823Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 591
    end_line: 715
apis:
  - protocol: file
    path: "osc.py#VrchatOscBridge.set_axes"
    description:
      zh: >
          同时设置多个摇杆轴。
          
      en: >
          Set several joystick axes at once.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge.set_axis"
    description:
      zh: >
          设置单个轴一段时间，之后自动停止。
          
      en: >
          Set one axis for a duration, then stop it automatically.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge.stop_axes"
    description:
      zh: >
          停止指定轴，或全部轴。
          
      en: >
          Stop the named axes, or all of them.
          
---
