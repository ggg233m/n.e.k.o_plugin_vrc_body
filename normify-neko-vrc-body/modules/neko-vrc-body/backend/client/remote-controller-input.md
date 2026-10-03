---
uid: 9e3c010f
id: neko-vrc-body.backend.client.remote-controller-input
parent: neko-vrc-body.backend.client
name: {zh: "虚拟控制器输入代理", en: "Virtual Controller Input Proxy"}
description:
  zh: >
      虚拟 Index 控制器输入的快速 IPC 代理。
      
  en: >
      The fast IPC proxy for the virtual Index controller input.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.582Z"
fingerprint: 8f6d5a4ca8fb37bed50fe66446dfcb2eb54d67a9caca0125f881ac24d9844a7e
source:
  - path: "backend/client.py"
    line: 572
    end_line: 626
apis:
  - protocol: file
    path: "backend/client.py#RemoteControllerInput.set_axes"
    description:
      zh: >
          写入虚拟 Index 控制器摇杆轴。
          
      en: >
          Writes the virtual controller stick axes.
          
  - protocol: file
    path: "backend/client.py#RemoteControllerInput.set_button"
    description:
      zh: >
          按下或释放一个虚拟控制器按键。
          
      en: >
          Presses or releases one virtual controller button.
          
  - protocol: file
    path: "backend/client.py#RemoteControllerInput.release"
    description:
      zh: >
          释放全部按住的虚拟输入。
          
      en: >
          Releases all held virtual inputs.
          
---
