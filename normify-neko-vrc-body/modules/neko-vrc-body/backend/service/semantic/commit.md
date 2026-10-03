---
uid: 9e3e0017
id: neko-vrc-body.backend.service.semantic.commit
parent: neko-vrc-body.backend.service.semantic
name: {zh: "语义提交", en: "Semantic Commit"}
description:
  zh: >
      把主 LLM 的回答提交回来，以及拒绝「回答了另一个请求」的绑定校验。
      
  en: >
      Committing the main LLM's answer back, and the binding check that refuses a commit answering a different request.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.754Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2466
    end_line: 2595
apis:
  - protocol: file
    path: "backend/service.py#BackendService.main_llm_semantic_commit"
    description:
      zh: >
          把主 LLM 的语义判断提交进世界状态存储。
          
      en: >
          Commit the main LLM's semantic judgement into the world store.
          
  - protocol: file
    path: "backend/service.py#BackendService._semantic_binding_matches"
    description:
      zh: >
          检查一次提交是否仍然对应它所回答的那个请求。
          
      en: >
          Check that a commit still refers to the request it answers.
          
---
