---
uid: 9e3a0204
id: neko-vrc-body.plugin.lifecycle.shutdown
parent: neko-vrc-body.plugin.lifecycle
name: {zh: "关闭", en: "Shutdown"}
description:
  zh: >
      on_shutdown：停止各桥、注销 agent 条目、释放输入并关闭后端。
      
  en: >
      on_shutdown: stop the bridges, unregister agent entries, release inputs and shut the backend down.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.876Z"
fingerprint: 1cbc358527e95295d9a81b577b9e5acab62572fe69134b706a03c84dfabcf05b
source:
  - path: "__init__.py"
    line: 491
    end_line: 508
apis:
  - protocol: file
    path: "__init__.py#NekoAnyadanceBodyPlugin.on_shutdown"
    description:
      zh: >
          按启动的逆序停止全部自有资源。
          
      en: >
          Stop every owned resource in reverse start-up order.
          
---
