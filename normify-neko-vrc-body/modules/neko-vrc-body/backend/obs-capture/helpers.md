---
uid: "9e410017"
id: neko-vrc-body.backend.obs-capture.helpers
parent: neko-vrc-body.backend.obs-capture
name: {zh: "可选依赖辅助", en: "Optional Dependency Helpers"}
description:
  zh: >
      可选依赖的惰性导入与受长度限制的错误文本。
      
  en: >
      Lazy import of optional dependencies plus length-bounded error text.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.688Z"
fingerprint: a425bb8c95520acbcda535ab5dc98f10420a4e78c156c4b6a9e006aceea078f9
source:
  - path: "backend/obs_capture.py"
    line: 34
    end_line: 42
apis:
  - protocol: file
    path: "backend/obs_capture.py#_optional_import"
    description:
      zh: >
          按需惰性导入可选模块，缺失时返回空结果而非报错。
          
      en: >
          Lazily imports an optional module on demand, yielding an empty result instead of an error when it is missing.
          
  - protocol: file
    path: "backend/obs_capture.py#_safe_error"
    description:
      zh: >
          把异常压成有长度上限的可展示文本。
          
      en: >
          Flattens an exception into displayable text with a length cap.
          
---
