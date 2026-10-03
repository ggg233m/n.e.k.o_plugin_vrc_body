---
uid: 9e3a0305
id: neko-vrc-body.body-kernel.protocol.encoding
parent: neko-vrc-body.body-kernel.protocol
name: {zh: "报文编码", en: "Payload Encoding"}
description:
  zh: >
      v1 报文布局与两个编码器，含专用于转向的只带头显帧。
      
  en: >
      The version-1 payload layout and the two encoders, including the head-only frame used purely for turning.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.788Z"
fingerprint: 4a5943879019db1637b9a60ce6e1045d4061f57b2413cf0c5466cce1190e304d
source:
  - path: "protocol.py"
    line: 65
    end_line: 129
apis:
  - protocol: file
    path: "protocol.py#frame_payload"
    description:
      zh: >
          把一帧渲染成 v1 报文映射。
          
      en: >
          Render a frame into the version-1 payload mapping.
          
  - protocol: file
    path: "protocol.py#encode_frame"
    description:
      zh: >
          把整帧编码成 UDP 报文。
          
      en: >
          Encode a full frame into a UDP packet.
          
  - protocol: file
    path: "protocol.py#encode_head_frame"
    description:
      zh: >
          编码只含 HMD 的帧；驱动会为省略的设备保留上一次位姿。
          
      en: >
          Encode an HMD-only frame; the driver keeps the previous pose for omitted devices.
          
deps:
  - kind: reference
    to: neko-vrc-body.body-kernel.model.frame
    label: {zh: "从帧中读设备", en: "Read devices from frame"}
  - kind: call
    to: neko-vrc-body.body-kernel.model.quat
    label: {zh: "净化四元数", en: "Sanitise quaternions"}
---
