---
uid: 9e40001b
id: neko-vrc-body.backend.perception.runtime.semantic-request
parent: neko-vrc-body.backend.perception.runtime
name: {zh: "主 LLM 语义请求", en: "Main LLM Semantic Request"}
description:
  zh: >
      按需向主 LLM 发起语义请求，并对选择器做归一化。
      
  en: >
      Issues on-demand semantic requests to the main LLM and normalizes the selectors.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.711Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 2920
    end_line: 3143
apis:
  - protocol: file
    path: "backend/vision.py#VisionRuntime.request_main_llm_semantics"
    description:
      zh: >
          向主 LLM 发起一次语义请求，去重与限流后交给语义工作线程。
          
      en: >
          Issues a semantic request to the main LLM, deduplicating and rate-limiting before handing it to the semantic worker.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime.main_llm_semantic_request"
    description:
      zh: >
          读取当前待处理的主 LLM 语义请求及其状态。
          
      en: >
          Reads the pending main-LLM semantic request and its status.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime._normalize_main_llm_selector"
    description:
      zh: >
          把外部传入的选择器归一化为内部可识别的对象选择集合。
          
      en: >
          Normalizes an externally supplied selector into an internally recognized set of object choices.
          
---
