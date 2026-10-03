---
uid: 9e3c0112
id: neko-vrc-body.backend.client.remote-vmc
parent: neko-vrc-body.backend.client
name: {zh: "VMC 代理", en: "VMC Proxies"}
description:
  zh: >
      VMC 中继与宿主 VMC 状态的代理。
      
  en: >
      The proxies for the VMC relay and the host VMC state.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.585Z"
fingerprint: 8f6d5a4ca8fb37bed50fe66446dfcb2eb54d67a9caca0125f881ac24d9844a7e
source:
  - path: "backend/client.py"
    line: 750
    end_line: 794
apis:
  - protocol: file
    path: "backend/client.py#RemoteVmcIdle.recalibrate"
    description:
      zh: >
          重新校准空闲的 VMC 中继代理。
          
      en: >
          Recalibrates the idle VMC relay proxy.
          
  - protocol: file
    path: "backend/client.py#RemoteHostVmc.snapshot"
    description:
      zh: >
          读取宿主 VMC 状态快照。
          
      en: >
          Reads a snapshot of the host VMC state.
          
---
