---
uid: 9e3f000c
id: neko-vrc-body.backend.autonomy.runtime.world
parent: neko-vrc-body.backend.autonomy.runtime
name: {zh: "运行时世界刷新", en: "Runtime World Refresh"}
description:
  zh: >
      用最新世界状态刷新目标可行性，并输出可读的状态快照。
      
  en: >
      Refreshes goal feasibility from the latest world state and emits a readable status snapshot.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.574Z"
fingerprint: 956f89ee80cd18326626bcfed51b616a94c813b4ec5dd9fa58499c2a52baf96a
source:
  - path: "backend/autonomy.py"
    line: 332
    end_line: 459
apis:
  - protocol: file
    path: "backend/autonomy.py#AutonomyRuntime.update_world"
    description:
      zh: >
          用最新世界状态刷新目标可行性。
          
      en: >
          Refreshes goal feasibility from the latest world state.
          
  - protocol: file
    path: "backend/autonomy.py#AutonomyRuntime.snapshot"
    description:
      zh: >
          输出可读的运行时状态快照。
          
      en: >
          Emits a readable snapshot of the runtime state.
          
---
