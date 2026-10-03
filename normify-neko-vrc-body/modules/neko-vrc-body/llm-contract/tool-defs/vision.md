---
uid: 9e3a0108
id: neko-vrc-body.llm-contract.tool-defs.vision
parent: neko-vrc-body.llm-contract.tool-defs
name: {zh: "视觉工具", en: "Vision Tools"}
description:
  zh: >
      启停感知与把最新帧作为图片部分取回的 Schema。
      
  en: >
      Schemas for starting/stopping perception and pulling the latest frame as an image part.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.840Z"
fingerprint: 348077cad7b4ebac68be08404f65a6094ebab38baa302e5e51c5b85edc242009
source:
  - path: "tool_defs.py"
    line: 439
    end_line: 492
apis:
  - protocol: file
    path: "tool_defs.py#VRC_VISION_START"
    description:
      zh: >
          工具 Schema：启动感知 worker。
          
      en: >
          Tool schema: start the perception worker.
          
  - protocol: file
    path: "tool_defs.py#VRC_VISION_STOP"
    description:
      zh: >
          工具 Schema：停止感知 worker。
          
      en: >
          Tool schema: stop the perception worker.
          
  - protocol: file
    path: "tool_defs.py#VRC_VISION_FRAME"
    description:
      zh: >
          工具 Schema：取最新帧作为图片部分返回。
          
      en: >
          Tool schema: fetch the latest frame as an image part.
          
---
