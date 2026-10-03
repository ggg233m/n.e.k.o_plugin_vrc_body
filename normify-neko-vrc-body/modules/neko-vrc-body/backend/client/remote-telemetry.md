---
uid: 9e3c0111
id: neko-vrc-body.backend.client.remote-telemetry
parent: neko-vrc-body.backend.client
name: {zh: "驱动遥测", en: "Driver Telemetry"}
description:
  zh: >
      驱动遥测快照的代理。
      
  en: >
      The proxy for driver telemetry snapshots.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.584Z"
fingerprint: 8f6d5a4ca8fb37bed50fe66446dfcb2eb54d67a9caca0125f881ac24d9844a7e
source:
  - path: "backend/client.py"
    line: 739
    end_line: 747
apis:
  - protocol: file
    path: "backend/client.py#RemoteDriverLog.snapshot"
    description:
      zh: >
          读取远端驱动日志的快照。
          
      en: >
          Reads a snapshot of the remote driver log.
          
---
