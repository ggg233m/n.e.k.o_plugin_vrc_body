---
uid: "9e500020"
id: neko-vrc-body.backend.nav-mapping.mapper.accumulator.acc-sync
parent: neko-vrc-body.backend.nav-mapping.mapper.accumulator
name: {zh: "累加器同步", en: "Accumulator Sync"}
description:
  zh: >
      分带票数累加器的同步、位姿平移与扩容。
      
  en: >
      Synchronisation, pose shifting and growth of the banded-vote accumulator.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.628Z"
fingerprint: caf6dd30bbd13ba420ba62eb770627a77f528a9314d72c2456ff0d2b6210709f
source:
  - path: "backend/nav_mapping.py"
    line: 427
    end_line: 562
apis:
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper._sync_acc"
    description:
      zh: >
          让分带票数累加器与当前位姿同步。
          
      en: >
          Synchronises the banded-vote accumulator with the current poses.
          
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper._grow_acc"
    description:
      zh: >
          扩容累加器以容纳新区域。
          
      en: >
          Grows the accumulator to fit a new region.
          
---
