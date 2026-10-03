---
uid: 9e3a021a
id: neko-vrc-body.plugin.agent.world-context-loop
parent: neko-vrc-body.plugin.agent
name: {zh: "世界上下文循环", en: "World Context Loop"}
description:
  zh: >
      长跑的异步循环：拉取世界增量、判定显著性，然后推送上下文或叫醒 agent。
      
  en: >
      The long-running async loop: poll world deltas, classify salience, and push context or wake the agent.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.856Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 1288
    end_line: 1482
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._world_context_loop"
    description:
      zh: >
          异步世界上下文轮询循环。
          
      en: >
          The async world-context polling loop.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._world_context_loop_run"
    description:
      zh: >
          一次迭代：拉取增量、判定显著性并推送上下文。
          
      en: >
          One iteration: poll the delta, classify it and push context.
          
---
