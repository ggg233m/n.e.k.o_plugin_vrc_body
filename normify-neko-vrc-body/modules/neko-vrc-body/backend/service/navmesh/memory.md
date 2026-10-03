---
uid: 9e3e0006
id: neko-vrc-body.backend.service.navmesh.memory
parent: neko-vrc-body.backend.service.navmesh
name: {zh: "Navmesh 记忆接口", en: "Navmesh Memory API"}
description:
  zh: >
      按世界组织的 navmesh 记忆管理面：列举、打标签、钉住、删除与配额清理。
      
  en: >
      Management surface over the per-world navmesh memory: listing, labelling, pinning, deletion and quota pruning.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.744Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 580
    end_line: 620
apis:
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_memory_summary"
    description:
      zh: >
          汇总记忆根目录。
          
      en: >
          Summarise the memory root.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_memory_sessions"
    description:
      zh: >
          列出某个世界下的会话。
          
      en: >
          List the sessions of one world.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_memory_session"
    description:
      zh: >
          读取一条会话记录。
          
      en: >
          Read one session record.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_memory_thumbnail"
    description:
      zh: >
          以图片字节读取某个关键帧的缩略图。
          
      en: >
          Read one keyframe thumbnail as image bytes.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_memory_update"
    description:
      zh: >
          重命名、钉住或取消钉住一个会话。
          
      en: >
          Relabel, pin or unpin a session.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_memory_delete"
    description:
      zh: >
          删除一个会话或整个世界。
          
      en: >
          Delete a session or a whole world.
          
  - protocol: file
    path: "backend/service.py#BackendService.navmesh_memory_prune"
    description:
      zh: >
          按配额清理记忆库。
          
      en: >
          Prune the store against its quota.
          
---
