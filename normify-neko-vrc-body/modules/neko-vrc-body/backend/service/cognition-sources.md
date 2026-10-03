---
uid: 9e3e000a
id: neko-vrc-body.backend.service.cognition-sources
parent: neko-vrc-body.backend.service
name: {zh: "认知数据源", en: "Cognition Sources"}
description:
  zh: >
      认知循环可读的有界数据源清单，以及阻止两个写入者争抢驱动的发送者冲突规则。
      
  en: >
      The bounded source list the cognition loop may read, and the sender-conflict rule that stops two writers fighting over the driver.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.734Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 750
    end_line: 820
apis:
  - protocol: file
    path: "backend/service.py#BackendService._cognition_sources"
    description:
      zh: >
          枚举认知循环可读取的有界数据源。
          
      en: >
          Enumerate the bounded data sources the cognition loop may read.
          
  - protocol: file
    path: "backend/service.py#BackendService._apply_driver_sender_conflict"
    description:
      zh: >
          报告插件与本机驱动之间的发送者冲突。
          
      en: >
          Report a sender conflict between the plugin and the local driver.
          
---
