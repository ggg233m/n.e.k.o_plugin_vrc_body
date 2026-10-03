---
uid: 9e3a0017
id: neko-vrc-body.body-kernel.frame-budget
parent: neko-vrc-body.body-kernel
name: {zh: "拉帧预算", en: "Frame Budget"}
description:
  zh: >
      拉图请求的滑动窗口限流器，时钟由调用方传入，因此窗口边界可被测试。
      
  en: >
      A sliding-window rate limiter for frame requests, with the caller's clock so the window edge is testable.
      
revision: c6cd0a7c9179d635671c54c1871c6d311878af48
updated_at: "2026-10-03T16:13:37.783Z"
fingerprint: 64cf562d05a9c1fbc558011c8d8da93a3bb26c950be64029b68bcf40de833f62
source:
  - path: "frame_budget.py"
    line: 20
    end_line: 96
apis:
  - protocol: file
    path: "frame_budget.py#FrameBudget.check"
    description:
      zh: >
          判定当前窗口内是否还能再服务一帧。
          
      en: >
          Decide whether one more frame may be served in this window.
          
  - protocol: file
    path: "frame_budget.py#FrameBudget.consume"
    description:
      zh: >
          消耗一个窗口预算单位。
          
      en: >
          Consume one unit of the window budget.
          
  - protocol: file
    path: "frame_budget.py#FrameBudget.status"
    description:
      zh: >
          报告当前窗口的上限、已用与剩余。
          
      en: >
          Report limit, used and remaining for the current window.
          
---
