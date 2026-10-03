---
uid: "9e430005"
id: neko-vrc-body.backend.nav-memory.store.deletion
parent: neko-vrc-body.backend.nav-memory.store
name: {zh: "删除与配额清理", en: "Deletion & Quota Pruning"}
description:
  zh: >
      删除与配额清理，并拒绝动到正在写入或被钉住的会话。
      
  en: >
      Deletes and prunes by quota, refusing to touch sessions that are being written or are pinned.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.640Z"
fingerprint: 8ac3ac2937f0825e7d07c1267ffa2e59000b81e0743552690708cb1b776bbb0e
source:
  - path: "backend/nav_memory.py"
    line: 412
    end_line: 472
apis:
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore.delete_session"
    description:
      zh: >
          删除单个会话及其数据。
          
      en: >
          Deletes a single session together with its data.
          
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore.delete_world"
    description:
      zh: >
          删除某个世界的整个记忆目录。
          
      en: >
          Deletes the whole memory directory of a world.
          
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore.prune"
    description:
      zh: >
          按配额裁掉最旧的会话。
          
      en: >
          Prunes the oldest sessions down to the quota.
          
  - protocol: file
    path: "backend/nav_memory.py#NavMemoryStore._refuse_active"
    description:
      zh: >
          在动某个会话前检查它是否受保护。
          
      en: >
          Checks whether a session is protected before touching it.
          
---
