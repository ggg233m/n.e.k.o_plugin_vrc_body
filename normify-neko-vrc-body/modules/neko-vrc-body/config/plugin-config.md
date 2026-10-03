---
uid: 9e3a0026
id: neko-vrc-body.config.plugin-config
parent: neko-vrc-body.config
name: {zh: "插件总配置", en: "PluginConfig"}
description:
  zh: >
      聚合配置对象与 from_mapping：畸形设置文件变成硬错误的地方只有这一处。
      
  en: >
      The aggregate config object and from_mapping, the single place where a malformed settings file becomes a hard error.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.814Z"
fingerprint: 55757bfaa420ee4e8c4479614d38026395bc79552cfe74aec0d249ef070ee23f
source:
  - path: "config.py"
    line: 450
    end_line: 1051
apis:
  - protocol: file
    path: "config.py#PluginConfig"
    description:
      zh: >
          交给每个子系统的聚合配置对象。
          
      en: >
          The aggregate configuration object handed to every subsystem.
          
  - protocol: file
    path: "config.py#PluginConfig.from_mapping"
    description:
      zh: >
          从原始设置映射构造 PluginConfig，任何越界都会抛错。
          
      en: >
          Build a PluginConfig from a raw settings mapping, raising on anything out of bounds.
          
---
