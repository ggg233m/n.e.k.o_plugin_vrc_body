---
uid: "9e410005"
id: neko-vrc-body.backend.local-perception.detector.onnx
parent: neko-vrc-body.backend.local-perception.detector
name: {zh: "ONNX Runtime 后端", en: "ONNX Runtime Backend"}
description:
  zh: >
      ONNX Runtime 后端：会话选项、CUDA 失败记录与推理调用。
      
  en: >
      The ONNX Runtime backend covering session options, CUDA failure recording and inference calls.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.611Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 764
    end_line: 905
apis:
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._initialize_onnxruntime"
    description:
      zh: >
          建立 ONNX Runtime 会话并按执行提供程序完成初始化。
          
      en: >
          Establishes the ONNX Runtime session and initializes it for the selected execution provider.
          
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._record_cuda_failure"
    description:
      zh: >
          记录 CUDA 初始化失败的原因以便状态如实上报。
          
      en: >
          Records why CUDA initialization failed so status can report it honestly.
          
---
