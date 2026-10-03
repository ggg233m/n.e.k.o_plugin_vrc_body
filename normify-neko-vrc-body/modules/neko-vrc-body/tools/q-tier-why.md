---
uid: 9e3a0030
id: neko-vrc-body.tools.q-tier-why
parent: neko-vrc-body.tools
name: {zh: "分层取证", en: "Quality Tier Forensics"}
description:
  zh: >
      诊断被降级/删除的障碍块各自票长什么样，用来分辨墙与噪点。
      
  en: >
      Diagnoses what the votes of each demoted obstacle block actually looked like, to tell walls from speckle.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.943Z"
fingerprint: 27842d95968507ba1ca77c7acc88dc7028316a74e14c61f10d8629789c000125
source:
  - path: "tools/q_tier_why.py"
    line: 23
    end_line: 108
apis:
  - protocol: file
    path: "tools/q_tier_why.py#main"
    description:
      zh: >
          报告每个被降级障碍块的票分布。
          
      en: >
          Report the vote distribution of each demoted obstacle block.
          
  - protocol: file
    path: "tools/q_tier_why.py#to_grid"
    description:
      zh: >
          把 mapper 累加器布局转换成 NavGrid 布局。
          
      en: >
          Convert the mapper accumulator layout into the NavGrid layout.
          
---
