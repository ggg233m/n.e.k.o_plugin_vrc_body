---
uid: 9e3a0218
id: neko-vrc-body.plugin.agent.semantic-request
parent: neko-vrc-body.plugin.agent
name: {zh: "语义请求", en: "Semantic Request"}
description:
  zh: >
      主 LLM 语义请求往返：渲染请求文本，并连同帧图片部分取回。
      
  en: >
      The main-LLM semantic request round trip: rendering the request text and fetching it with its frame image part.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.851Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 1066
    end_line: 1188
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._semantic_request_text"
    description:
      zh: >
          把待处理的语义请求渲染成提示文本。
          
      en: >
          Render the pending semantic request as prompt text.
          
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin._fetch_semantic_request_parts"
    description:
      zh: >
          取回待处理的语义请求及其帧图片部分。
          
      en: >
          Fetch the pending semantic request together with its frame image part.
          
---
