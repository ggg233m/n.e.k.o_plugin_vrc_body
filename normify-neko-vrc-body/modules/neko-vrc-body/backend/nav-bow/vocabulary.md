---
uid: 9e3d0208
id: neko-vrc-body.backend.nav-bow.vocabulary
parent: neko-vrc-body.backend.nav-bow
name: {zh: "词袋词汇树", en: "BOW Vocabulary"}
description:
  zh: >
      汉明词汇树：按位多数投票求质心、汉明 k-means 分裂，以及位平面打包。
      
  en: >
      Hamming vocabulary tree: bitwise-majority centroids, Hamming k-means splitting, and bit-plane packing.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.617Z"
fingerprint: 9abec6334f072858dc90a006b03915ada2a5e6afa259dfee7295722f80cd2d0e
source:
  - path: "backend/nav_bow.py"
    line: 45
    end_line: 156
apis:
  - protocol: file
    path: "backend/nav_bow.py#BowVocabulary.train"
    description:
      zh: >
          按位多数投票求质心并用汉明 k-means 分裂训练词汇。
          
      en: >
          Trains the vocabulary by bitwise majority centroid and Hamming k-means splitting.
          
  - protocol: file
    path: "backend/nav_bow.py#BowVocabulary.transform"
    description:
      zh: >
          把描述子向量量化为词汇词。
          
      en: >
          Quantizes a descriptor vector into vocabulary words.
          
  - protocol: file
    path: "backend/nav_bow.py#BowVocabulary.save"
    description:
      zh: >
          把训练好的词汇树序列化到磁盘。
          
      en: >
          Serializes the trained vocabulary to disk.
          
  - protocol: file
    path: "backend/nav_bow.py#bits"
    description:
      zh: >
          把向量按位平面打包为整数。
          
      en: >
          Encodes a vector into a bit-plane packed word.
          
---
