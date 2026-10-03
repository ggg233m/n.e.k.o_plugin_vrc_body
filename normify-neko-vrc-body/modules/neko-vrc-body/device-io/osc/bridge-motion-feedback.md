---
uid: 9e3a034a
id: neko-vrc-body.device-io.osc.bridge-motion-feedback
parent: neko-vrc-body.device-io.osc
name: {zh: "动作反馈", en: "Motion Feedback"}
description:
  zh: >
      从速度流推出的动作反馈 —— 导航器闭环所依赖的信号。
      
  en: >
      Motion feedback derived from the velocity stream — the signal the navigator closes its loop on.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.825Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 905
    end_line: 1073
apis:
  - protocol: file
    path: "osc.py#VrchatOscBridge.motion_feedback"
    description:
      zh: >
          读取近期化身移动，拒绝超过给定年龄的样本。
          
      en: >
          Read recent avatar movement, refusing samples older than the given age.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge.motion_history"
    description:
      zh: >
          读取有界的近期速度历史。
          
      en: >
          Read the bounded recent velocity history.
          
---
