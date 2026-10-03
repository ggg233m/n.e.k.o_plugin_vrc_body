---
uid: "9e410020"
id: neko-vrc-body.backend.webui.config-store
parent: neko-vrc-body.backend.webui
name: {zh: "配置存储", en: "Config Store"}
description:
  zh: >
      把面板改动校验后写进覆盖文件，并报告哪些改动需要重启。
      
  en: >
      Validates panel edits into the override file and reports which changes require a restart.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.767Z"
fingerprint: c7f821962592b220a5fd4321e8dd682dc62fa77be2922e76864bbc5cea56a4b5
source:
  - path: "backend/webui.py"
    line: 21
    end_line: 220
apis:
  - protocol: file
    path: "backend/webui.py#StandaloneConfigStore.save"
    description:
      zh: >
          校验后持久化面板改动到覆盖配置文件。
          
      en: >
          Validates and persists panel edits into the override config file.
          
  - protocol: file
    path: "backend/webui.py#StandaloneConfigStore.snapshot"
    description:
      zh: >
          返回当前生效配置的只读快照。
          
      en: >
          Returns a read-only snapshot of the currently effective configuration.
          
  - protocol: file
    path: "backend/webui.py#deep_merge"
    description:
      zh: >
          把增量配置递归合并进基线配置。
          
      en: >
          Recursively merges a partial configuration into the baseline configuration.
          
  - protocol: file
    path: "backend/webui.py#_validate_update"
    description:
      zh: >
          校验更新载荷只触碰可编辑字段并标记需重启的改动。
          
      en: >
          Validates that an update payload only touches editable fields and flags restart-requiring changes.
          
---
