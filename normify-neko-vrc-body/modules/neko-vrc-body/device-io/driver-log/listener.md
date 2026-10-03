---
uid: 9e3a0361
id: neko-vrc-body.device-io.driver-log.listener
parent: neko-vrc-body.device-io.driver-log
name: {zh: "遥测监听器", en: "Telemetry Listener"}
description:
  zh: >
      加入组播组、跟踪发送者、序列去重，并汇总驱动报告的内容。
      
  en: >
      Joining the multicast group, tracking senders, de-duplicating sequences and summarising what the driver reports.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.819Z"
fingerprint: c8a4fd5ddeaad285f69e1b67a07314be7c0738f57a94deaa19af7b24210179e9
source:
  - path: "driver_log.py"
    line: 243
    end_line: 542
apis:
  - protocol: file
    path: "driver_log.py#DriverLogListener.start"
    description:
      zh: >
          加入驱动的组播遥测组。
          
      en: >
          Join the driver's multicast telemetry group.
          
  - protocol: file
    path: "driver_log.py#DriverLogListener.ingest_packet"
    description:
      zh: >
          接收一个数据报并跟踪其发送者。
          
      en: >
          Ingest one packet and track its sender.
          
  - protocol: file
    path: "driver_log.py#DriverLogListener.snapshot"
    description:
      zh: >
          汇总驱动当前报告的内容。
          
      en: >
          Summarise what the driver currently reports.
          
---
