---
uid: 9e43000a
id: neko-vrc-body.backend.nav-xsession.index.build
parent: neko-vrc-body.backend.nav-xsession.index
name: {zh: "索引构建与缓存", en: "Index Build & Cache"}
description:
  zh: >
      扫描世界目录重建索引，并原子写回缓存；永远不抛，失败只返回信息。
      
  en: >
      Scans the world directory to rebuild the index and writes the cache back atomically; it never raises and only reports failures.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.658Z"
fingerprint: bb74d4fa731d16faee2a784a6247828d8c29ded78a95d4fda7cd69a1ce17ddba
source:
  - path: "backend/nav_xsession.py"
    line: 144
    end_line: 341
apis:
  - protocol: file
    path: "backend/nav_xsession.py#build_index"
    description:
      zh: >
          扫描历史会话以重建世界索引。
          
      en: >
          Scans past sessions to rebuild a world index.
          
  - protocol: file
    path: "backend/nav_xsession.py#load_index"
    description:
      zh: >
          读取缓存的世界索引。
          
      en: >
          Loads the cached world index.
          
  - protocol: file
    path: "backend/nav_xsession.py#_save_index"
    description:
      zh: >
          原子写回索引缓存文件。
          
      en: >
          Atomically writes the index cache file back.
          
---
