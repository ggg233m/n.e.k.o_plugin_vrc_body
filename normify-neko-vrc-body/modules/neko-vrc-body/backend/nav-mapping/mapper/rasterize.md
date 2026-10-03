---
uid: "9e420028"
id: neko-vrc-body.backend.nav-mapping.mapper.rasterize
parent: neko-vrc-body.backend.nav-mapping.mapper
name: {zh: "栅格化与质量分层", en: "Rasterize and quality tiers"}
description:
  zh: >
      把累加器栅格化，并按质量分层决定一个格到底是空地还是障碍。
      
  en: >
      Rasterizes the accumulator and uses quality tiers to decide whether a cell is free space or an obstacle.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.632Z"
fingerprint: caf6dd30bbd13ba420ba62eb770627a77f528a9314d72c2456ff0d2b6210709f
source:
  - path: "backend/nav_mapping.py"
    line: 780
    end_line: 901
apis:
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper.rasterize"
    description:
      zh: >
          把累加器栅格化成栅格图。
          
      en: >
          Rasterizes the accumulator into a grid.
          
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper._by_quality"
    description:
      zh: >
          按质量分层对格子分组。
          
      en: >
          Groups cells by their quality tier.
          
  - protocol: file
    path: "backend/nav_mapping.py#KeyframeGridMapper.coverage_counts"
    description:
      zh: >
          统计当前栅格的覆盖情况。
          
      en: >
          Counts the coverage of the current grid.
          
---
