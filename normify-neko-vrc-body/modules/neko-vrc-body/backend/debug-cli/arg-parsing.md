---
uid: 9e3e0022
id: neko-vrc-body.backend.debug-cli.arg-parsing
parent: neko-vrc-body.backend.debug-cli
name: {zh: "参数解析", en: "Argument Parsing"}
description:
  zh: >
      调试命令行的参数解析：来自字面量或文件的 JSON，以及裸 JSON 标量。
      
  en: >
      Argument parsing for the debug CLI: JSON from a literal or a file, and bare JSON scalars.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.590Z"
fingerprint: ea1de34602a67e7be6b4c263a343c3d70a69e31f189f484adadc872b211d42a8
source:
  - path: "backend/debug_cli.py"
    line: 14
    end_line: 36
apis:
  - protocol: file
    path: "backend/debug_cli.py#_json_arg"
    description:
      zh: >
          从字面量或文件读取一个 JSON 参数。
          
      en: >
          Read a JSON argument from a literal or a file.
          
  - protocol: file
    path: "backend/debug_cli.py#_scalar_arg"
    description:
      zh: >
          解析 OSC 参数命令所用的 JSON 标量。
          
      en: >
          Parse a JSON scalar used by an OSC parameter command.
          
---
