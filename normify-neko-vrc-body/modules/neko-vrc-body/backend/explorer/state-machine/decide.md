---
uid: 9e50001f
id: neko-vrc-body.backend.explorer.state-machine.decide
parent: neko-vrc-body.backend.explorer.state-machine
name: {zh: "探索决策", en: "Explore Decision"}
description:
  zh: >
      扫描、短前进、再扫描的决策与执行记录。
      
  en: >
      The scan, short advance, rescan decision and the record of what was applied.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.599Z"
fingerprint: 3ceb238589b7465b0375c659e6db1f5f984144bfe544a81d29d28423b1db0ad9
source:
  - path: "backend/explorer.py"
    line: 190
    end_line: 298
apis:
  - protocol: file
    path: "backend/explorer.py#ExplorerStateMachine.decide"
    description:
      zh: >
          决定扫描、短前进、再扫描。
          
      en: >
          Decides scan, short advance, then scan again.
          
  - protocol: file
    path: "backend/explorer.py#ExplorerStateMachine.record_applied"
    description:
      zh: >
          记录已执行的探索指令。
          
      en: >
          Records the directive that was applied.
          
---
