---
uid: 9e3a0349
id: neko-vrc-body.device-io.osc.bridge-snapshot
parent: neko-vrc-body.device-io.osc
name: {zh: "桥接快照", en: "Bridge Snapshot"}
description:
  zh: >
      读取完整桥接状态，含化身参数缓存与任何错误。
      
  en: >
      Reading the full bridge state, including the avatar parameter cache and any errors.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.827Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 860
    end_line: 903
apis:
  - protocol: file
    path: "osc.py#VrchatOscBridge.snapshot"
    description:
      zh: >
          读取完整桥接快照，可选择包含全部参数。
          
      en: >
          Read the full bridge snapshot, optionally including every parameter.
          
---
