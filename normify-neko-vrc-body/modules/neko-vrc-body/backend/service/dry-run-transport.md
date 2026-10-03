---
uid: 9e3e0002
id: neko-vrc-body.backend.service.dry-run-transport
parent: neko-vrc-body.backend.service
name: {zh: "空跑传输层", en: "Dry-run Transport"}
description:
  zh: >
      只计数而不真正发送的传输层，使整条身体链路可以在没有驱动时运行。
      
  en: >
      A transport that counts frames without sending anything, so the whole body path can run without a driver.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.735Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 213
    end_line: 225
apis:
  - protocol: file
    path: "backend/service.py#_DryRunDatagramTransport"
    description:
      zh: >
          只统计帧数、绝不触碰网络的调度器传输层。
          
      en: >
          A scheduler transport that only counts frames and never touches the network.
          
---
