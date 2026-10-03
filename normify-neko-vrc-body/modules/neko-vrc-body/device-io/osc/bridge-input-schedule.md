---
uid: 9e3a0345
id: neko-vrc-body.device-io.osc.bridge-input-schedule
parent: neko-vrc-body.device-io.osc
name: {zh: "输入调度", en: "Input Scheduling"}
description:
  zh: >
      延迟输入队列：调度的脉冲、延时释放，以及如何安全取消。
      
  en: >
      The deferred input queue: scheduled pulses, delayed releases and how to cancel them safely.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.824Z"
fingerprint: 3fcb3a92c0aef32c1b9a0c1176ca166a975663c66bc5f25f298ec6877d8cc3b2
source:
  - path: "osc.py"
    line: 474
    end_line: 588
apis:
  - protocol: file
    path: "osc.py#VrchatOscBridge.schedule_input_pulse"
    description:
      zh: >
          调度一次按下再释放的脉冲。
          
      en: >
          Schedule a press-then-release pulse.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge.schedule_button_release"
    description:
      zh: >
          在延时后调度一次按键释放。
          
      en: >
          Schedule a button release after a delay.
          
  - protocol: file
    path: "osc.py#VrchatOscBridge.cancel_scheduled_inputs"
    description:
      zh: >
          取消待执行的脉冲，可选择先释放已按下的输入。
          
      en: >
          Cancel pending pulses, optionally releasing held inputs first.
          
---
