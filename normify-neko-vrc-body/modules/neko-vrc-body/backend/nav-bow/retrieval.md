---
uid: 9e3d0209
id: neko-vrc-body.backend.nav-bow.retrieval
parent: neko-vrc-body.backend.nav-bow
name: {zh: "词袋检索索引", en: "BOW Retrieval Index"}
description:
  zh: >
      按关键帧累积的词直方图加 idf 加权，打分用直方图交集除以较小的 L1 范数。
      
  en: >
      Per-keyframe word histograms with idf weighting, scored by histogram intersection over the smaller L1 norm.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.617Z"
fingerprint: 9abec6334f072858dc90a006b03915ada2a5e6afa259dfee7295722f80cd2d0e
source:
  - path: "backend/nav_bow.py"
    line: 159
    end_line: 237
apis:
  - protocol: file
    path: "backend/nav_bow.py#BowIndex.add"
    description:
      zh: >
          把某个关键帧的描述子加入索引。
          
      en: >
          Add one keyframe's descriptors to the index.
          
  - protocol: file
    path: "backend/nav_bow.py#BowIndex.add_words"
    description:
      zh: >
          为某个关键帧加入已转换的词。
          
      en: >
          Add already-transformed words for one keyframe.
          
  - protocol: file
    path: "backend/nav_bow.py#BowIndex.query"
    description:
      zh: >
          为一次查询检索外观最相似的关键帧。
          
      en: >
          Retrieve the most appearance-similar keyframes for a query.
          
  - protocol: file
    path: "backend/nav_bow.py#BowIndex.query_words"
    description:
      zh: >
          直接用预先算好的词进行检索。
          
      en: >
          Retrieve directly from pre-computed words.
          
---
