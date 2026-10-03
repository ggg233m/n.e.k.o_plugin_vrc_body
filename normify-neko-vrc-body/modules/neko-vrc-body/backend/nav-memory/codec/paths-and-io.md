---
uid: "9e700009"
id: neko-vrc-body.backend.nav-memory.codec.paths-and-io
parent: neko-vrc-body.backend.nav-memory.codec
name: {zh: "路径与读写", en: "Paths and IO"}
description:
  zh: >
      模式版本、安全的按世界目录命名，以及其余一切文件写入都经过的原子读写助手。
      
  en: >
      Schema version, safe per-world directory naming, and the atomic write/read helpers every other file write goes through.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.638Z"
fingerprint: 8ac3ac2937f0825e7d07c1267ffa2e59000b81e0743552690708cb1b776bbb0e
source:
  - path: "backend/nav_memory.py"
    line: 40
    end_line: 93
apis:
  - protocol: file
    path: "backend/nav_memory.py#world_dir_name"
    description:
      zh: >
          安全的按世界目录名加原键的短哈希。
          
      en: >
          Safe per-world directory name plus a short hash of the original key.
          
  - protocol: file
    path: "backend/nav_memory.py#_write_atomic"
    description:
      zh: >
          原子写入助手。
          
      en: >
          Atomic write helper.
          
---
