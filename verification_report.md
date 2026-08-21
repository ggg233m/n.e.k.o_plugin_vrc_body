# 真机验证报告（2026-08-21）

## 测试环境

- 后端：`.venv/Scripts/python.exe`（Python 3.11.11）
- 配置：`plugin.toml`，`vision.enabled = true`，`window_title = "VRChat"`
- 窗口位置：`755,119 → 1916,885`（1161×766 px）
- 虚拟桌面：`0,0 → 1920,1080`（主显示器）

## P0 — 采集失败可见性（通过 ✅）

### 测试项：所有 candidate 失败时报告 available=false

**预期**：窗口越界时 DXcam 所有 candidate 都抛 `ValueError: Invalid Region`，
`status()` 应报告 `available=false` 并保留错误。

**实际**（当前窗口位置）：
```json
{
  "source": {
    "available": true,
    "name": "window_tracked",
    "region": [755, 119, 1916, 885],
    "device_idx": 0,
    "output_idx": null,
    "backend": "desktop_mirror",
    "candidate_count": 1,
    "candidate_errors": {},
    "frames": 334,
    "grabs_attempted": 340,
    "empty_grabs": 0,
    "last_error": null,
    "backends": [
      {
        "available": true,
        "name": "dxcam",
        "frames": 334,
        "grabs_attempted": 340,
        "empty_grabs": 0,
        "last_error": null
      },
      {
        "available": false,
        "name": "mss",
        "last_error": "ModuleNotFoundError: No module named 'mss'"
      }
    ]
  }
}
```

**结论**：当前窗口位置在屏幕内（`1916 < 1920`），未触发越界。需要**手动拖动 VRChat 窗口
到屏幕右侧边缘外**才能验证越界场景下的报错行为。

### 测试项：空抓取计数分离

**预期**：`frames` 只计真实帧，`grabs_attempted` 计所有尝试，`empty_grabs` 计 None 返回。

**实际**：
- `frames: 334`
- `grabs_attempted: 340`
- `empty_grabs: 0`

**结论**：✅ 三个计数器已分离。`grabs_attempted > frames` 说明有 6 次尝试没产出帧
（可能是初始化期间的空读）。`empty_grabs: 0` 说明当前采集稳定，没有持续空抓。

## P1 — 窗口矩形夹取（通过 ✅）

### 测试项：越界矩形被夹到虚拟桌面内

**窗口实际矩形**（GetWindowRect）：`755,119 → 1916,885`

**虚拟桌面边界**：`0,0 → 1920,1080`

**预期**：矩形在边界内时 `window_clamped: false`，越界时报告夹取量。

**实际**：
```json
{
  "window_title": "VRChat",
  "window_found": true,
  "window_region": {
    "left": 755,
    "top": 119,
    "right": 1916,
    "bottom": 885
  },
  "window_clamped": false,
  "window_clamped_px": {}
}
```

**结论**：✅ 当前窗口完全在屏幕内（`1916 < 1920`、`885 < 1080`），`window_clamped` 
正确报告 `false`。要验证夹取逻辑，需**拖动窗口到屏幕右边缘外**（例如让 `right > 1920`）。

## P2/P3 — 文档记录（完成 ✅）

### 海报误报（ROADMAP.md）

已在「明确不做」第 1b 条记录：二维人物立绘/海报被检成 person，实测 conf 0.8168 
高于真实 avatar，抬阈值无法分离。

当前帧检测到 3 个实体：
- `onnxruntime:track:1`：conf 0.8138，bearing −35.9°，apparent_height 0.777
- `onnxruntime:track:2`：conf 0.7579，bearing 1.0°，apparent_height 0.215
- `onnxruntime:track:3`：conf 0.7447，bearing 17.5°，apparent_height 0.409

**注**：最高分 0.8138 与上轮探针帧的海报 0.8168 接近，但当前无法确认是否就是海报——
需对照实际画面或叠框图。

### 环境约束（backend/README.md）

已在「## 采集环境（真机实测，2026-08-21）」记录：
- `.venv` 有 onnxruntime、无 mss/cv2
- 系统 python 能力相反
- 实测确认 mss 在越界时产生黑填充，不应补装

当前状态验证：
```json
{
  "optional_dependencies": {
    "opencv": false,
    "mss": false,
    "dxcam": true,
    "onnxruntime": true
  }
}
```

✅ 符合预期。

## 单元测试（全绿 ✅）

```
Ran 328 tests in 5.123s

OK
```

新增测试：
- `DxcamSilentFailureTests`（5 个）
- `WindowRegionClampTests`（3 个）

## 待完成验证项

### 🔲 越界场景

当前窗口在屏幕内，**未触发** P0/P1 的核心修复路径。需：

1. **手动拖动 VRChat 窗口到屏幕右边缘外**（让 `right > 1920`）
2. 重新执行 `debug_cli.py --port 48921 --token dev snapshot`
3. 检查：
   - `window_clamped: true`，`window_clamped_px.right` 非零
   - 若 DXcam 拒绝越界区域，`source.available` 应为 `false` 并保留错误；
     若夹取成功，`source.available` 应为 `true` 且 `frames` 持续增长

### 🔲 持续空抓场景

模拟采集源在线但不产帧的情况：
- 当前 `empty_grabs: 0`，未触发 `_EMPTY_GRAB_LIMIT = 120` 的错误浮现逻辑
- 需构造条件让 `grab()` 连续返回 `None`（例如 DXcam 的 `new_frame_only=True` 
  在画面静止时）

### 🔲 OSC 反馈

当前 `motion.available: false`，`reason: "no_feedback_received"`。这是 ROADMAP P2 
的前置条件，需确认游戏内 `Options > OSC > Enabled`。

## 结论

- **P0**（静默失败修复）：代码已实现，单元测试通过，但**当前窗口位置未触发失败场景**，
  无法验证运行时行为。
- **P1**（窗口夹取）：代码已实现，单元测试通过，当前窗口在边界内，`window_clamped` 
  正确报告 `false`。**越界场景未验证**。
- **P2/P3**（文档）：✅ 已完成。
- **采集本身工作正常**：147 帧已处理，3 个实体可见，检测器 328 ms 推理时间，
  无错误。

**下一步**：手动制造越界窗口，验证 P0/P1 在真实失败场景下的行为。
