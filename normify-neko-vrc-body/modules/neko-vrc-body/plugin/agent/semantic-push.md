---
uid: 9e3a0219
id: neko-vrc-body.plugin.agent.semantic-push
parent: neko-vrc-body.plugin.agent
name: {zh: "语义被动推送", en: "Passive Semantic Push"}
description:
  zh: >
      把语义请求被动推进宿主对话，含取消替换与拒绝记账。
      
  en: >
      Passive semantic pushes into the host conversation, including cancellation replacement and rejection bookkeeping.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.850Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 1190
    end_line: 1286
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._push_passive_semantic_parts"
    description:
      zh: >
          把非阻塞的语义请求推进宿主对话。
          
      en: >
          Push a non-blocking semantic request into the host conversation.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._replace_cancelled_semantic_push"
    description:
      zh: >
          替换掉被宿主报告为已取消的语义推送。
          
      en: >
          Replace a semantic push that the host reported as cancelled.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._semantic_push_was_submitted"
    description:
      zh: >
          检查语义推送是否真的被提交。
          
      en: >
          Check whether a semantic push was actually submitted.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._record_semantic_push_rejection"
    description:
      zh: >
          记录语义推送被拒绝的原因。
          
      en: >
          Record why a semantic push was rejected.
          
---
