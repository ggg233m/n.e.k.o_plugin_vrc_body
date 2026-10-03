---
uid: 9e3a034b
id: neko-vrc-body.device-io.osc.bridge-awareness
parent: neko-vrc-body.device-io.osc
name: {zh: "化身 Awareness", en: "Avatar Awareness"}
description:
  zh: >
      从收到的 OSC 参数推导出化身 aware 状态。
      
  en: >
      Deriving the avatar awareness state from the received OSC parameters.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.823Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 1075
    end_line: 1114
apis:
  - protocol: file
    path: "osc.py#VrchatOscBridge.awareness"
    description:
      zh: >
          读取由接收路径驱动的化身 aware 状态。
          
      en: >
          Read the avatar awareness state driven by the receive path.
          
---
