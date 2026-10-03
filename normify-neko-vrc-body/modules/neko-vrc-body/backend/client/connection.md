---
uid: 9e3c0109
id: neko-vrc-body.backend.client.connection
parent: neko-vrc-body.backend.client
name: {zh: "连接生命周期", en: "Connection Lifecycle"}
description:
  zh: >
      子进程启动、回环端口分配、stderr 排空与连接关闭。
      
  en: >
      Subprocess startup, loopback port allocation, stderr draining and connection teardown.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.580Z"
fingerprint: 8f6d5a4ca8fb37bed50fe66446dfcb2eb54d67a9caca0125f881ac24d9844a7e
source:
  - path: "backend/client.py"
    line: 49
    end_line: 238
apis:
  - protocol: file
    path: "backend/client.py#BackendClient.start"
    description:
      zh: >
          启动后端子进程并建立回环连接。
          
      en: >
          Starts the backend subprocess and establishes the loopback connection.
          
  - protocol: file
    path: "backend/client.py#BackendClient.stop"
    description:
      zh: >
          关闭连接并终止后端子进程。
          
      en: >
          Closes the connection and terminates the backend subprocess.
          
---
