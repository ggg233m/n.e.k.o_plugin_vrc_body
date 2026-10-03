---
uid: 9e3a0045
id: neko-vrc-body.backend.reid-embedder
parent: neko-vrc-body.backend
name: {zh: "OSNet 重识别嵌入器", en: "OSNet Re-ID Embedder"}
description:
  zh: >
      OSNet ONNX 外观嵌入器：加载失败自行记录，绝不抛出。
      
  en: >
      An OSNet ONNX appearance embedder that records its own load failures and never raises.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.726Z"
fingerprint: 6e436740029cefd580d12b92aa448edff9b8902cc50d79ec36d0a7b513610a38
source:
  - path: "backend/reid_embedder.py"
    line: 31
    end_line: 220
apis:
  - protocol: file
    path: "backend/reid_embedder.py#OsnetReidEmbedder.embed"
    description:
      zh: >
          把化身裁剪图嵌入成学习到的外观向量，失败则说明原因。
          
      en: >
          Embed an avatar crop into a learned appearance vector, or report why not.
          
  - protocol: file
    path: "backend/reid_embedder.py#OsnetReidEmbedder.status"
    description:
      zh: >
          报告模型加载状态，绝不抛异常。
          
      en: >
          Report model load state without ever raising.
          
---
