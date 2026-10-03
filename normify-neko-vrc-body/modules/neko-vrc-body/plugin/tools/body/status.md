---
uid: 9e3a022e
id: neko-vrc-body.plugin.tools.body.status
parent: neko-vrc-body.plugin.tools.body
name: {zh: "身体状态", en: "Body Status"}
description:
  zh: >
      读取实时身体快照：调度器状态、当前动作、aware 以及真正被请求的小节。
      
  en: >
      Reading the live body snapshot: scheduler state, current action, awareness and the sections actually asked for.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.892Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2666
    end_line: 2736
apis:
  - protocol: rpc
    path: "body_status"
    description:
      zh: >
          工具：读取身体运行时快照，可按小节裁剪。
          
      en: >
          Tool: read the body runtime snapshot, optionally narrowed by section.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._body_snapshot"
    description:
      zh: >
          构建状态工具回传的身体快照。
          
      en: >
          Build the body snapshot handed back by the status tool.
          
---
