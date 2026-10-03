---
uid: 9e3a0106
id: neko-vrc-body.llm-contract.tool-defs.semantic
parent: neko-vrc-body.llm-contract.tool-defs
name: {zh: "语义提交工具", en: "Semantic Commit Tool"}
description:
  zh: >
      主 LLM 把自己对场景的语义判读提交进世界状态存储的 Schema。
      
  en: >
      The schema through which the main LLM commits its own semantic reading of the scene into the world store.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.840Z"
fingerprint: 348077cad7b4ebac68be08404f65a6094ebab38baa302e5e51c5b85edc242009
source:
  - path: "tool_defs.py"
    line: 297
    end_line: 354
apis:
  - protocol: file
    path: "tool_defs.py#VRC_SEMANTIC_COMMIT"
    description:
      zh: >
          工具 Schema：把 VLM 对实体的语义判断提交进世界状态存储。
          
      en: >
          Tool schema: commit a VLM semantic judgement about entities into the world store.
          
---
