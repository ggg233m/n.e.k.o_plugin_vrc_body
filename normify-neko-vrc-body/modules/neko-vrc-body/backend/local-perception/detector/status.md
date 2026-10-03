---
uid: "9e410008"
id: neko-vrc-body.backend.local-perception.detector.status
parent: neko-vrc-body.backend.local-perception.detector
name: {zh: "运行状态报告", en: "Runtime Status Reporting"}
description:
  zh: >
      报告实际使用的后端、设备、线程数与输出布局，而非配置意图。
      
  en: >
      Reports the backend, device, thread count and output layout actually in use rather than the configured intent.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.613Z"
fingerprint: 79c9a433737d5a577a65099bc58ef16b459f9d0f0e3e032043d50d8952cfd22e
source:
  - path: "backend/local_perception.py"
    line: 1140
    end_line: 1276
apis:
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector.status"
    description:
      zh: >
          对外给出检测器当前真实生效的运行状态。
          
      en: >
          Exposes the detector's actually effective runtime state.
          
  - protocol: file
    path: "backend/local_perception.py#OpenVinoLocalDetector._row_widths"
    description:
      zh: >
          探测模型输出每一行的字段宽度以确定布局。
          
      en: >
          Probes the field width of each output row to determine the layout.
          
---
