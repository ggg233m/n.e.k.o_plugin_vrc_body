---
uid: "9e700010"
id: neko-vrc-body.backend.navigator.lifecycle.open-close
parent: neko-vrc-body.backend.navigator.lifecycle
name: {zh: "开启与关闭", en: "Open and Close"}
description:
  zh: >
      局部导航线程的构造与启停。
      
  en: >
      Construction and start/stop of the local navigation thread.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.677Z"
fingerprint: 7b0d865aeb8117d58a082330ef160ea039413daf419a608b12a24c863990a93d
source:
  - path: "backend/navigator.py"
    line: 387
    end_line: 530
apis:
  - protocol: file
    path: "backend/navigator.py#LocalNavigator.start"
    description:
      zh: >
          启动局部导航线程。
          
      en: >
          Start the local navigation thread.
          
  - protocol: file
    path: "backend/navigator.py#LocalNavigator.stop"
    description:
      zh: >
          停止局部导航线程。
          
      en: >
          Stop the local navigation thread.
          
---
