---
uid: "9e430007"
id: neko-vrc-body.backend.nav-xsession.config
parent: neko-vrc-body.backend.nav-xsession
name: {zh: "跨会话检索配置", en: "Cross-Session Retrieval Config"}
description:
  zh: >
      跨会话检索的配置、词汇树路径解析与内容哈希。
      
  en: >
      Configuration of cross-session retrieval, vocabulary-tree path resolution and content hashing.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.655Z"
fingerprint: bb74d4fa731d16faee2a784a6247828d8c29ded78a95d4fda7cd69a1ce17ddba
source:
  - path: "backend/nav_xsession.py"
    line: 51
    end_line: 83
apis:
  - protocol: file
    path: "backend/nav_xsession.py#XSessionConfig"
    description:
      zh: >
          跨会话检索的配置数据类。
          
      en: >
          Configuration dataclass of cross-session retrieval.
          
  - protocol: file
    path: "backend/nav_xsession.py#_vocab_path"
    description:
      zh: >
          解析词汇树文件路径。
          
      en: >
          Resolves the vocabulary tree file path.
          
  - protocol: file
    path: "backend/nav_xsession.py#_sha1"
    description:
      zh: >
          计算内容哈希用于缓存失效判断。
          
      en: >
          Computes a content hash used to detect cache invalidation.
          
---
