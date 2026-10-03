---
uid: 9e3a0024
id: neko-vrc-body.config.navmesh-sections
parent: neko-vrc-body.config
name: {zh: "导航网格配置段", en: "Navmesh Sections"}
description:
  zh: >
      在线建图配置与按世界分区的关键帧记忆库，含唯一真值源 world_scale。
      
  en: >
      The online mapping configuration and the per-world keyframe memory store, including the single world_scale truth source.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.813Z"
fingerprint: 55757bfaa420ee4e8c4479614d38026395bc79552cfe74aec0d249ef070ee23f
source:
  - path: "config.py"
    line: 290
    end_line: 354
apis:
  - protocol: file
    path: "config.py#NavmeshConfig"
    description:
      zh: >
          在线 navmesh 建图调参；world_scale 是米制的唯一真值源。
          
      en: >
          Online navmesh mapping tuning; world_scale is the single truth source for metres.
          
  - protocol: file
    path: "config.py#NavmeshMemoryConfig"
    description:
      zh: >
          按世界分区的关键帧记忆：位姿、ORB 特征与缩略图，存放在状态目录下。
          
      en: >
          Per-world keyframe memory: poses, ORB features and thumbnails under the state directory.
          
---
