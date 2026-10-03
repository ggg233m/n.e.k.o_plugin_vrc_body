---
uid: 9e3a0340
id: neko-vrc-body.device-io.osc.codec
parent: neko-vrc-body.device-io.osc
name: {zh: "OSC 编解码", en: "OSC Codec"}
description:
  zh: >
      OSC 1.0 报文编码与地址、参数校验，只用 VRChat 需要的类型。
      
  en: >
      OSC 1.0 message encoding, address and parameter validation, using only the types VRChat needs.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.829Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 21
    end_line: 152
apis:
  - protocol: file
    path: "osc.py#encode_osc_message"
    description:
      zh: >
          编码一条 OSC 1.0 报文。
          
      en: >
          Encode one OSC 1.0 message.
          
  - protocol: file
    path: "osc.py#validate_parameter_name"
    description:
      zh: >
          校验并归一化化身参数名。
          
      en: >
          Validate and normalise an avatar parameter name.
          
  - protocol: file
    path: "osc.py#normalize_parameter_value"
    description:
      zh: >
          把参数值归一化成 bool、int 或 float。
          
      en: >
          Normalise a parameter value into bool, int or float.
          
deps:
  - kind: call
    to: neko-vrc-body.device-io.osc.decoder
    label: {zh: "复用解码规则", en: "Reuse decoder rules"}
---
