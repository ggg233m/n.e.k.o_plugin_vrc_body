---
uid: 9e3a0343
id: neko-vrc-body.device-io.osc.bridge-parameter
parent: neko-vrc-body.device-io.osc
name: {zh: "参数与聊天框", en: "Parameters and Chatbox"}
description:
  zh: >
      化身参数写入与聊天框输入 —— 工具用得最多的两条出站路径。
      
  en: >
      Avatar parameter writes and chatbox typing, the two outbound paths the tools use most.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.826Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 385
    end_line: 399
apis:
  - protocol: file
    path: "osc.py#VrchatOscBridge.send_parameter"
    description:
      zh: >
          通过 OSC 设置化身表情或参数。
          
      en: >
          Set an avatar expression or parameter over OSC.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge.send_chatbox"
    description:
      zh: >
          通过 OSC 在 VRChat 聊天框输入一行文本。
          
      en: >
          Type a line into the VRChat chatbox over OSC.
          
---
