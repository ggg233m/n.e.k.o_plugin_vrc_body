---
uid: "9e430010"
id: neko-vrc-body.backend.nav-xsession.tracker.init
parent: neko-vrc-body.backend.nav-xsession.tracker
name: {zh: "追踪器初始化与后台装载", en: "Tracker Init & Background Load"}
description:
  zh: >
      会话侧追踪器：后台线程装载世界索引，主线程不因此阻塞。
      
  en: >
      The session-side tracker loads the world index on a background thread so the main thread never blocks.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.660Z"
fingerprint: bb74d4fa731d16faee2a784a6247828d8c29ded78a95d4fda7cd69a1ce17ddba
source:
  - path: "backend/nav_xsession.py"
    line: 645
    end_line: 711
apis:
  - protocol: file
    path: "backend/nav_xsession.py#XSessionTracker.wait_ready"
    description:
      zh: >
          等待后台索引装载完成。
          
      en: >
          Waits for the background index load to finish.
          
  - protocol: file
    path: "backend/nav_xsession.py#XSessionTracker._load_bg"
    description:
      zh: >
          在后台线程装载世界索引。
          
      en: >
          Loads the world index on a background thread.
          
---
