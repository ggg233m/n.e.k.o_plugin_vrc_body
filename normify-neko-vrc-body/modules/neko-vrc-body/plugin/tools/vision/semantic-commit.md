---
uid: 9e3a0235
id: neko-vrc-body.plugin.tools.vision.semantic-commit
parent: neko-vrc-body.plugin.tools.vision
name: {zh: "语义提交", en: "Semantic Commit"}
description:
  zh: >
      把主 LLM 自己对一帧的语义判读提交进后端世界状态存储。
      
  en: >
      Committing the main LLM's own semantic reading of a frame into the backend world store.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.898Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 2971
    end_line: 3008
apis:
  - protocol: rpc
    path: "vrc_semantic_commit"
    description:
      zh: >
          工具：把对场景的语义判断提交进世界状态存储。
          
      en: >
          Tool: commit a semantic judgement of the scene into the world store.
          
---
