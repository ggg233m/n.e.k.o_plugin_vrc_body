---
uid: 9e3c0110
id: neko-vrc-body.backend.client.remote-autonomy
parent: neko-vrc-body.backend.client
name: {zh: "自主代理", en: "Autonomy Agent Proxy"}
description:
  zh: >
      授权与目标的代理；启用授权必须显式执行。
      
  en: >
      The arming and goal proxy; enabling the autonomy agent always requires an explicit call.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.581Z"
fingerprint: 8f6d5a4ca8fb37bed50fe66446dfcb2eb54d67a9caca0125f881ac24d9844a7e
source:
  - path: "backend/client.py"
    line: 629
    end_line: 736
apis:
  - protocol: file
    path: "backend/client.py#RemoteAutonomy.arm"
    description:
      zh: >
          启用远端自主代理。
          
      en: >
          Arms the remote autonomy agent.
          
  - protocol: file
    path: "backend/client.py#RemoteAutonomy.goal"
    description:
      zh: >
          设置远端自主代理的目标。
          
      en: >
          Sets the goal of the remote autonomy agent.
          
  - protocol: file
    path: "backend/client.py#RemoteAutonomy.wander_step"
    description:
      zh: >
          驱动远端自主代理单步游走。
          
      en: >
          Drives one wander step of the remote autonomy agent.
          
  - protocol: file
    path: "backend/client.py#RemoteAutonomy.intent"
    description:
      zh: >
          向远端自主代理提交一条意图。
          
      en: >
          Submits an intent to the remote autonomy agent.
          
---
