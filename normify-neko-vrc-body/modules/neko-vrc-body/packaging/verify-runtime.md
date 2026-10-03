---
uid: 9e3a0036
id: neko-vrc-body.packaging.verify-runtime
parent: neko-vrc-body.packaging
name: {zh: "运行时验证", en: "Runtime Verifier"}
description:
  zh: >
      用干净的 Python 检查安装后的依赖、跑一次真实模型推理并拉起后端服务。
      
  en: >
      Uses a clean Python to check the installed dependencies, run a real model inference and start the backend service.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.843Z"
fingerprint: 2aa2718b8e55daa8971b1bb8e09bd276e94cb0327fad7c41128adfb20c7b3443
source:
  - path: "packaging/verify_runtime.py"
    line: 18
    end_line: 83
apis:
  - protocol: file
    path: "packaging/verify_runtime.py#main"
    description:
      zh: >
          检查安装后的依赖、跑一次真实推理并探测后端服务。
          
      en: >
          Check installed dependencies, run a real inference and probe the backend service.
          
---
