---
uid: 9e3a0341
id: neko-vrc-body.device-io.osc.decoder
parent: neko-vrc-body.device-io.osc
name: {zh: "OSC 解码器", en: "OSC Decoder"}
description:
  zh: >
      报文解码：字符串读取、类型解包、消息提取与嵌套 bundle 展平。
      
  en: >
      Packet decoding: string reads, type unpacking, message extraction and nested bundle flattening.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.830Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 155
    end_line: 239
apis:
  - protocol: file
    path: "osc.py#decode_osc_packet"
    description:
      zh: >
          解码 OSC 报文，处理 bundle 与嵌套。
          
      en: >
          Decode an OSC packet, handling bundles and nesting.
          
  - protocol: file
    path: "osc.py#_INPUT_ADDRESSES"
    description:
      zh: >
          桥接愿意接收的地址集合。
          
      en: >
          Addresses the bridge is willing to receive on.
          
---
