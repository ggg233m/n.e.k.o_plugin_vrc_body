---
uid: 9e40001c
id: neko-vrc-body.backend.perception.runtime.semantic-commit
parent: neko-vrc-body.backend.perception.runtime
name: {zh: "语义结果并入", en: "Semantic Result Commit"}
description:
  zh: >
      把主 LLM 的回答并入观测，并消费已处理的路由请求。
      
  en: >
      Merges the main LLM's answer into the observation and consumes already-processed route requests.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.710Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 3145
    end_line: 3406
apis:
  - protocol: file
    path: "backend/vision.py#VisionRuntime.commit_main_llm_semantics"
    description:
      zh: >
          把主 LLM 的回答校验后并入当前观测的语义字段。
          
      en: >
          Validates the main LLM's answer and merges it into the current observation's semantic fields.
          
  - protocol: file
    path: "backend/vision.py#VisionRuntime._process_semantic_job"
    description:
      zh: >
          处理一个完成的语义任务，把结果与路由请求写回运行时。
          
      en: >
          Processes a finished semantic job, writing the result and route request back into the runtime.
          
---
