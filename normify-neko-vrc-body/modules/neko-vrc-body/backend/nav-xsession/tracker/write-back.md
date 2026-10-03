---
uid: "9e430012"
id: neko-vrc-body.backend.nav-xsession.tracker.write-back
parent: neko-vrc-body.backend.nav-xsession.tracker
name: {zh: "约束写回与状态", en: "Write-Back & Status"}
description:
  zh: >
      会话末写回约束并刷新索引；任何失败都不抛，跨会话检索只是候选来源。
      
  en: >
      Writes constraints back and refreshes the index at session end; it never raises, since cross-session retrieval is only a source of candidates.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.661Z"
fingerprint: bb74d4fa731d16faee2a784a6247828d8c29ded78a95d4fda7cd69a1ce17ddba
source:
  - path: "backend/nav_xsession.py"
    line: 804
    end_line: 882
apis:
  - protocol: file
    path: "backend/nav_xsession.py#XSessionTracker.write_back"
    description:
      zh: >
          在会话末把约束写回记忆库。
          
      en: >
          Writes the constraints back into the memory store at session end.
          
  - protocol: file
    path: "backend/nav_xsession.py#XSessionTracker.status"
    description:
      zh: >
          返回追踪器状态，含已确认的约束数量。
          
      en: >
          Returns tracker status including the number of confirmed constraints.
          
---
