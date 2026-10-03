---
uid: "9e430006"
id: neko-vrc-body.backend.nav-memory.store.paths
parent: neko-vrc-body.backend.nav-memory.store
name: {zh: "目录遍历与路径解析", en: "Path Traversal & Resolution"}
description:
  zh: >
      目录遍历与路径解析，会话 id 经正则校验后才能拼进路径。
      
  en: >
      Directory traversal and path resolution, where a session id is regex-validated before being joined into a path.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.641Z"
fingerprint: 8ac3ac2937f0825e7d07c1267ffa2e59000b81e0743552690708cb1b776bbb0e
source:
  - path: "backend/nav_memory.py"
    line: 474
    end_line: 512
apis:
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore._world"
    description:
      zh: >
          解析某个世界实例的记忆目录。
          
      en: >
          Resolves the memory directory of a world instance.
          
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore._session"
    description:
      zh: >
          解析某个世界下会话的目录。
          
      en: >
          Resolves the session directory under a world.
          
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore._session_info"
    description:
      zh: >
          读取并解析某个会话的元信息。
          
      en: >
          Reads and parses a session's metadata.
          
---
