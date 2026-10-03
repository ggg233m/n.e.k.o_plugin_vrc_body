---
uid: 9e3e0013
id: neko-vrc-body.backend.service.world-model
parent: neko-vrc-body.backend.service
name: {zh: "世界模型与时间轴", en: "World Model and Timeline"}
description:
  zh: >
      手动世界身份接口，以及由宿主驱动的动作时间轴控制。
      
  en: >
      The manual world-identity surface and the action timeline controls the host drives.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.762Z"
fingerprint: 2a79b646c07d727b5a4af07ee54d27f94676f029fc37d65837d89ff97727e163
source:
  - path: "backend/service.py"
    line: 2097
    end_line: 2143
apis:
  - protocol: file
    path: "backend/service.py#BackendService.world_model_status"
    description:
      zh: >
          读取世界模型状态。
          
      en: >
          Read the world model status.
          
  - protocol: file
    path: "backend/service.py#BackendService.world_model_start"
    description:
      zh: >
          启动世界模型。
          
      en: >
          Start the world model.
          
  - protocol: file
    path: "backend/service.py#BackendService.world_model_set_world"
    description:
      zh: >
          手动设置世界身份。
          
      en: >
          Set the world identity manually.
          
  - protocol: file
    path: "backend/service.py#BackendService.action_timeline_status"
    description:
      zh: >
          读取动作时间轴状态。
          
      en: >
          Read the action timeline status.
          
  - protocol: file
    path: "backend/service.py#BackendService.action_timeline_begin"
    description:
      zh: >
          打开动作时间轴。
          
      en: >
          Open the action timeline.
          
---
