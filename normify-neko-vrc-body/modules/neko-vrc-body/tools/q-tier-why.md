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
      
revision: 0eb991034d815aafdf6ad8342f413fecb385d5b5
updated_at: "2026-10-16T21:51:59.430Z"
fingerprint: bcb6a0eab7c33f53cab1d4ba81da07ed060fc78748c90324759caf9eb91d93b5
source:
  - path: "research/tools/q_tier_why.py"
    line: 23
    end_line: 108
apis:
  - protocol: file
    path: "research/tools/q_tier_why.py#main"
    description:
      zh: >
          报告每个被降级障碍块的票分布。
          
      en: >
          Report the vote distribution of each demoted obstacle block.
          
  - protocol: file
    path: "research/tools/q_tier_why.py#to_grid"
    description:
      zh: >
          把 mapper 累加器布局转换成 NavGrid 布局。
          
      en: >
          Convert the mapper accumulator layout into the NavGrid layout.
          
---
