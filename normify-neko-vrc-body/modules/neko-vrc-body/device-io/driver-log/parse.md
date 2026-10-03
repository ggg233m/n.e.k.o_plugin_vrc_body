---
uid: 9e3a0360
id: neko-vrc-body.device-io.driver-log.parse
parent: neko-vrc-body.device-io.driver-log
name: {zh: "遥测解析", en: "Telemetry Parsing"}
description:
  zh: >
      防御性地解码一个遥测数据报：任何畸形或缺失字段都降级为中性值而非抛错。
      
  en: >
      Decoding one telemetry datagram defensively: every malformed or missing field degrades to a neutral value rather than raising.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.820Z"
fingerprint: c8a4fd5ddeaad285f69e1b67a07314be7c0738f57a94deaa19af7b24210179e9
source:
  - path: "driver_log.py"
    line: 28
    end_line: 240
apis:
  - protocol: file
    path: "driver_log.py#parse_driver_log_event"
    description:
      zh: >
          解码一个遥测数据报，不可用时返回 None。
          
      en: >
          Decode one telemetry datagram, or return None if unusable.
          
  - protocol: file
    path: "driver_log.py#_parse_action_fields"
    description:
      zh: >
          解析动作时间轴字段，任何缺失字段都降级为中性值。
          
      en: >
          Parse action timeline fields, degrading every missing field to a neutral value.
          
  - protocol: file
    path: "driver_log.py#DRIVER_LOG_PROTOCOL_VERSION"
    description:
      zh: >
          受支持的遥测协议版本。
          
      en: >
          The supported telemetry protocol version.
          
---
