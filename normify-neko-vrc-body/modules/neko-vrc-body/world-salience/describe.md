---
uid: 9e3a002c
id: neko-vrc-body.world-salience.describe
parent: neko-vrc-body.world-salience
name: {zh: "实体描述", en: "Entity Description"}
description:
  zh: >
      世界文本里的实体描述行，补上了 agent 需要的距离档与方位档。
      
  en: >
      Human-readable entity lines for the world text, now carrying the distance and bearing bands the agent needs.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.945Z"
fingerprint: 9514e5be7514a04afbe83acf4af23398f6939c9ad2d179fbcb3a12d4af685844
source:
  - path: "world_salience.py"
    line: 232
    end_line: 257
apis:
  - protocol: file
    path: "world_salience.py#describe_entities"
    description:
      zh: >
          渲染世界上下文里的实体描述行，带上距离档与方位档。
          
      en: >
          Render entity lines for the world context text, carrying distance and bearing bands.
          
---
