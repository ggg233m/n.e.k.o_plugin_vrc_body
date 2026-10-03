---
uid: 9e3d020b
id: neko-vrc-body.backend.nav-loop.features
parent: neko-vrc-body.backend.nav-loop
name: {zh: "特征提取与匹配", en: "Feature Extraction"}
description:
  zh: >
      左目 ORB 特征提取到头部 base 系，以及带比值检验的互相匹配。
      
  en: >
      Left-eye ORB feature extraction into the head base frame, plus mutual matching with a ratio test.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.625Z"
fingerprint: 0d562774afcbca2772a8abb5b0d193fdfb369fdf0074750d5fb80d71774492c7
source:
  - path: "backend/nav_loop.py"
    line: 92
    end_line: 123
apis:
  - protocol: file
    path: "backend/nav_loop.py#extract_features"
    description:
      zh: >
          从左目图像提取 ORB 特征并变换到头部 base 系。
          
      en: >
          Extracts ORB features from the left eye and transforms them into the head base frame.
          
  - protocol: file
    path: "backend/nav_loop.py#_match_mutual"
    description:
      zh: >
          带比值检验的互相匹配过滤特征对应。
          
      en: >
          Filters feature correspondences by mutual matching with a ratio test.
          
---
