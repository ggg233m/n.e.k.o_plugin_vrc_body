---
uid: 9e3a0344
id: neko-vrc-body.device-io.osc.bridge-input
parent: neko-vrc-body.device-io.osc
name: {zh: "虚拟输入", en: "Virtual Input"}
description:
  zh: >
      虚拟输入动作与按键，含把保持时长收敛到窗口内，避免卡住的按下变成卡住的化身。
      
  en: >
      Virtual input actions and buttons, including the clamped hold window that keeps a stuck press from becoming a stuck avatar.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.825Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 401
    end_line: 472
apis:
  - protocol: file
    path: "osc.py#VrchatOscBridge.send_input"
    description:
      zh: >
          按下或释放一个虚拟输入动作。
          
      en: >
          Press or release a virtual input action.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge.send_button"
    description:
      zh: >
          按下或释放一个虚拟按键。
          
      en: >
          Press or release a virtual button.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge.pulse_input"
    description:
      zh: >
          在收敛后的保持窗口内按下并释放一个输入。
          
      en: >
          Press and release an input within a clamped hold window.
          
---
