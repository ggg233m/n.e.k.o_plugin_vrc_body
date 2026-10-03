---
uid: 9e3e0003
id: neko-vrc-body.backend.service.boot
parent: neko-vrc-body.backend.service
name: {zh: "构造装配", en: "Construction"}
description:
  zh: >
      构造：把配置、调度器、OSC 桥、VMC 中继、世界存储、认知、自主与导航器接在一起。
      
  en: >
      Construction: wires the config, scheduler, OSC bridge, VMC relay, world store, cognition, autonomy and navigator together.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.732Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 231
    end_line: 480
apis:
  - protocol: file
    path: "backend/service.py#BackendService.__init__"
    description:
      zh: >
          构造服务持有的全部长期资源。
          
      en: >
          Construct every long-lived resource the service owns.
          
---
