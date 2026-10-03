---
uid: "9e500001"
id: neko-vrc-body.backend.local-perception.detector.init.openvino-compile
parent: neko-vrc-body.backend.local-perception.detector.init
name: {zh: "OpenVINO 编译", en: "OpenVINO Compilation"}
description:
  zh: >
      设备候选探测与 OpenVINO 编译，按优先级逐个尝试并报告实际做到了什么。
      
  en: >
      Device-candidate probing and OpenVINO compilation, trying each option in priority order and reporting what was actually achieved.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.610Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 554
    end_line: 762
apis:
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._compile_openvino"
    description:
      zh: >
          在某个 OpenVINO 设备候选上编译模型。
          
      en: >
          Compiles the model on one OpenVINO device candidate.
          
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._compile_on_device"
    description:
      zh: >
          直接在当前运行的设备上编译模型。
          
      en: >
          Compiles the model straight on the running device.
          
---
