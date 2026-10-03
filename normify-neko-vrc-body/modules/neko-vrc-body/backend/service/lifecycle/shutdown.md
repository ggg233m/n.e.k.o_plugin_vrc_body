---
uid: 9e3e000d
id: neko-vrc-body.backend.service.lifecycle.shutdown
parent: neko-vrc-body.backend.service.lifecycle
name: {zh: "服务关闭", en: "Service Shutdown"}
description:
  zh: >
      按逆序停止并释放全部按住的输入，使停机时不会有东西卡在按下状态。
      
  en: >
      Stopping in reverse order and releasing every held input, so nothing is left stuck on during shutdown.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.737Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 1034
    end_line: 1088
apis:
  - protocol: file
    path: "backend/service.py#BackendService.stop"
    description:
      zh: >
          按启动的逆序停止全部自有资源。
          
      en: >
          Stop every owned resource in reverse start-up order.
          
  - protocol: file
    path: "backend/service.py#BackendService._release_all_inputs"
    description:
      zh: >
          释放全部仍然按住的虚拟控制器输入。
          
      en: >
          Release every held virtual controller input.
          
---
