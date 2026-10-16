---
uid: 9e3a002e
id: neko-vrc-body.tools.q-tier-ab
parent: neko-vrc-body.tools
name: {zh: "质量分层 A/B", en: "Quality Tier A/B"}
description:
  zh: >
      拿真实录制跑在线 mapper 本体，对比扁平计数与质量分层计数的差异。
      
  en: >
      Runs the real online mapper over real recordings to compare flat vote counting against quality-tiered counting.
      
revision: 0eb991034d815aafdf6ad8342f413fecb385d5b5
updated_at: "2026-10-16T21:51:59.429Z"
fingerprint: 6066353185a0ea85820c32a6b0f1fdcd4c329ed5823be98f3947d5d2b9ff1ab9
source:
  - path: "research/tools/q_tier_ab.py"
    line: 32
    end_line: 305
apis:
  - protocol: file
    path: "research/tools/q_tier_ab.py#replay"
    description:
      zh: >
          把录制回放进真实在线 mapper，按环距报告走廊障碍格数量。
          
      en: >
          Replay a recording into the real online mapper and report corridor blockers per ring.
          
  - protocol: file
    path: "research/tools/q_tier_ab.py#components"
    description:
      zh: >
          按大小给 baseline 障碍连通块分组，用来分辨墙与噪点。
          
      en: >
          Group baseline obstacle blocks by size to tell walls from speckle.
          
  - protocol: file
    path: "research/tools/q_tier_ab.py#main"
    description:
      zh: >
          对全部录制跑 A/B 并渲染改动前后的栅格图。
          
      en: >
          Run the A/B over every recording and render before/after grids.
          
---
