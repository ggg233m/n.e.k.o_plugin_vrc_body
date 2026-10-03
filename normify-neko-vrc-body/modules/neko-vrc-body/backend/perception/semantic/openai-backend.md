---
uid: "9e400010"
id: neko-vrc-body.backend.perception.semantic.openai-backend
parent: neko-vrc-body.backend.perception.semantic
name: {zh: "OpenAI 兼容语义后端", en: "OpenAI-Compatible Semantic Backend"}
description:
  zh: >
      按变化触发的结构化 VLM 适配器，每分钟最多调用三十次。
      
  en: >
      A change-triggered structured VLM adapter, limited to at most thirty calls per minute.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.719Z"
fingerprint: 17e492b70dcdcd4a0900c09e0386d276eb07bbcbf2e92c7a6ee7e4f4a3df2737
source:
  - path: "backend/vision.py"
    line: 1573
    end_line: 1748
apis:
  - protocol: file
    path: "backend/vision.py#OpenAICompatibleSemanticBackend.observe"
    description:
      zh: >
          把图像与上一帧变化判断后按需发起一次 VLM 判读。
          
      en: >
          Triggers one VLM interpretation on demand after judging image change against the previous frame.
          
  - protocol: file
    path: "backend/vision.py#OpenAICompatibleSemanticBackend._call"
    description:
      zh: >
          向兼容接口发起一次结构化判读请求并解析其响应。
          
      en: >
          Issues one structured interpretation request to the compatible endpoint and parses the response.
          
---
