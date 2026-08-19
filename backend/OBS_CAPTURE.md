# OBS 画面桥接（可选）

当 DXGI Desktop Duplication 和 MSS 在本机返回 `BitBlt failed` 或
`0x80070005` 时，可以让 OBS 负责捕获 VRChat 桌面镜像，再把画面交给本后端。
这不是权限提升，也不会读取 SteamVR 的隐藏眼睛纹理。

## 首选：OBS Virtual Camera

1. 在 OBS 中添加 VRChat 窗口/显示器捕获，确认预览画面正常。
2. 点击 **Start Virtual Camera**。
3. 后端选择 `ObsVirtualCameraFrameSource`，指定 `camera_index`（`-1` 会探测
   前 8 个设备）。Windows 通常优先使用 DirectShow；也可显式指定 `dshow`
   或 `msmf`。
4. 安装可选依赖：`opencv-python`。它不会成为插件的硬依赖。

该适配器有一个独立读取线程，只保留最新帧；视觉模型慢时会丢旧帧，不会积压。
它的延迟取决于 OBS、摄像头驱动和模型，不能保证零延迟。`status()` 中的
`frames`、`last_frame_age_ms`、`read_failures` 和 `candidate_errors` 可用于诊断。

## 后备：obs-websocket v5

当不能使用虚拟摄像头时，可在 OBS 中启用 obs-websocket v5，并选择一个包含
VRChat 捕获的场景或源名，使用 `ObsWebSocketFrameSource`：

```python
ObsWebSocketFrameSource(
    source_name="VRChat Mirror",
    host="127.0.0.1",
    port=4455,
    password=os.environ.get("OBS_WEBSOCKET_PASSWORD"),
)
```

它调用 `GetSourceScreenshot`，每次 `read()` 只取得一张最新截图，依赖可选的
`websocket-client` 和 `opencv-python`（或 Pillow）解码。截图编码和 websocket
往返会增加延迟，因此不建议作为高频导航主链路；它更适合观察模式或虚拟摄像头
不可用时的兼容回退。密码不会出现在 `status()` 中。

## 安全与故障行为

- OBS 未运行、虚拟摄像头未启动、源名错误或依赖缺失时，适配器返回
  `available=false` 和明确的 `last_error`，不会伪造帧。
- 采集源只提供图像；世界实体和玩家/聊天持久化策略仍由后端视觉与世界状态层
  执行。
- 采集不可用时，导航器应保持停止；不要把“没有帧”解释成“场景为空”。
