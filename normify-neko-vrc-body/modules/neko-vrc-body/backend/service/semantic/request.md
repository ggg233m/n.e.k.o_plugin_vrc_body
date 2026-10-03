---
uid: 9e3e0016
id: neko-vrc-body.backend.service.semantic.request
parent: neko-vrc-body.backend.service.semantic
name: {zh: "语义请求", en: "Semantic Request"}
description:
  zh: >
      把待处理的语义请求交给主 LLM，并附上它必须解析的选择器与限流状态。
      
  en: >
      Handing the pending semantic request to the main LLM, with the selector it must resolve and any rate limit state.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.755Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2405
    end_line: 2464
apis:
  - protocol: file
    path: "backend/service.py#BackendService.main_llm_semantic_request"
    description:
      zh: >
          读取留给主 LLM 的待处理语义请求。
          
      en: >
          Read the pending semantic request for the main LLM.
          
---
