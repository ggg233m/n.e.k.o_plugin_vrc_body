---
uid: 9e3a0382
id: neko-vrc-body.clips.nya.library
parent: neko-vrc-body.clips.nya
name: {zh: "片段库", en: "Clip Library"}
description:
  zh: >
      磁盘片段库：安全路径解析、基于签名的缓存、编目与意图驱动的选择。
      
  en: >
      The on-disk clip library: safe path resolution, signature-based caching, cataloguing and intent-driven selection.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.807Z"
fingerprint: af36750f4baf9f8ec574db149233d6e34f52a4cca5324558b7d87e202a6a0414
source:
  - path: "nya.py"
    line: 190
    end_line: 411
apis:
  - protocol: file
    path: "nya.py#ClipLibrary.load"
    description:
      zh: >
          按名加载片段，带基于签名的缓存。
          
      en: >
          Load a clip by name, with a signature-based cache.
          
  - protocol: file
    path: "nya.py#ClipLibrary.select_for_intent"
    description:
      zh: >
          为一个语义意图确定性地挑选片段。
          
      en: >
          Choose a clip deterministically for a semantic intent.
          
  - protocol: file
    path: "nya.py#ClipLibrary._metrics"
    description:
      zh: >
          报告片段库中已索引与未索引的片段数量。
          
      en: >
          Report the library's indexed and unindexed clip counts.
          
---
