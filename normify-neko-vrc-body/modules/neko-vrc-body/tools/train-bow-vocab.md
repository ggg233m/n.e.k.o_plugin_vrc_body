---
uid: 9e3a0031
id: neko-vrc-body.tools.train-bow-vocab
parent: neko-vrc-body.tools
name: {zh: "词袋词汇训练", en: "BOW Vocabulary Trainer"}
description:
  zh: >
      训练 ORB 词袋词汇树，为回环检测提供基于外观的候选生成器。
      
  en: >
      Trains the ORB bag-of-words vocabulary tree that gives loop closure its appearance-based candidate generator.
      
revision: 0eb991034d815aafdf6ad8342f413fecb385d5b5
updated_at: "2026-10-16T21:51:59.431Z"
fingerprint: ed275c95575bd6cfe8cf326754049a6422152067b51f628f50c9d647d67ed4db
source:
  - path: "research/tools/train_bow_vocab.py"
    line: 29
    end_line: 102
apis:
  - protocol: file
    path: "research/tools/train_bow_vocab.py#main"
    description:
      zh: >
          训练并保存 ORB 词袋词汇树。
          
      en: >
          Train the ORB bag-of-words vocabulary tree and save it.
          
  - protocol: file
    path: "research/tools/train_bow_vocab.py#load_descriptors"
    description:
      zh: >
          从录制里加载到指定比例为止的 ORB 描述子。
          
      en: >
          Load ORB descriptors from a recording up to a fraction of the run.
          
---
