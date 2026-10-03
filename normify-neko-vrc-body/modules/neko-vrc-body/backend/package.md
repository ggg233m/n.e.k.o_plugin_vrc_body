---
uid: 9e3a0041
id: neko-vrc-body.backend.package
parent: neko-vrc-body.backend
name: {zh: "后端包边界", en: "Backend Package Boundary"}
description:
  zh: >
      独立运行时的边界，刻意不导入插件 SDK，使后端可单独运行。
      
  en: >
      The standalone runtime boundary, deliberately free of any plugin SDK import so the backend can run on its own.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.692Z"
fingerprint: 20dc0e38a7a347f024c8517066494a17fd42cde1f066f3022aca79d1681d4da9
source:
  - path: "backend/__init__.py"
    line: 1
    end_line: 7
apis:
  - protocol: file
    path: "backend/__init__.py"
    description:
      zh: >
          后端包的运行时边界，刻意不导入任何 N.E.K.O 插件 SDK。
          
      en: >
          The backend package boundary, deliberately free of any N.E.K.O plugin SDK import.
          
---
