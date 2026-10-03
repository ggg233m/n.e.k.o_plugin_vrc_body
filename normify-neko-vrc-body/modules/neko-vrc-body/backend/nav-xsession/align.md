---
uid: 9e43000e
id: neko-vrc-body.backend.nav-xsession.align
parent: neko-vrc-body.backend.nav-xsession
name: {zh: "新会话对齐", en: "New Session Alignment"}
description:
  zh: >
      把新会话对齐进旧会话世界系；自动采纳会挑确认最多的旧会话，够门槛才对齐。
      
  en: >
      Aligns a new session into the old session's world frame; auto-adoption picks the most confirmed old session and only aligns once the threshold is met.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.655Z"
fingerprint: bb74d4fa731d16faee2a784a6247828d8c29ded78a95d4fda7cd69a1ce17ddba
source:
  - path: "backend/nav_xsession.py"
    line: 508
    end_line: 642
apis:
  - protocol: file
    path: "backend/nav_xsession.py#align_into"
    description:
      zh: >
          用给定规范把新会话变换进旧世界系。
          
      en: >
          Transforms the new session into the old world frame with a given gauge.
          
  - protocol: file
    path: "backend/nav_xsession.py#align_into_auto"
    description:
      zh: >
          自动挑选旧会话并在对齐后合并索引。
          
      en: >
          Auto-selects an old session and merges the index after aligning.
          
---
