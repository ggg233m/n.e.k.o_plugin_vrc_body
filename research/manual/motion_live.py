#!/usr/bin/env python
"""独立测试脚本：验证 VRChat 内置参数回传与 motion_feedback()。

不依赖后端进程，直接创建 OSC 桥并每秒打印实测速度。用于确认：
1. VRChat 是否回传了 VelocityX/Y/Z、Grounded、Upright、AngularY
2. motion_feedback() 的 available 状态与 reason
3. 站立/移动时的实测速度值

运行方式：
    python test_motion_live.py

按 Ctrl+C 退出。
"""

from __future__ import annotations

import sys
import time

from neko_anyadance_body.config import VrchatOscConfig
from neko_anyadance_body.osc import VrchatOscBridge


def main() -> int:
    print("创建 OSC 桥 (send=127.0.0.1:9000, listen=127.0.0.1:9001)...")
    config = VrchatOscConfig(
        enabled=True,
        send_host="127.0.0.1",
        send_port=9000,
        listen_host="127.0.0.1",
        listen_port=9001,
        allowed_sender="127.0.0.1",
        awareness_parameters=(
            "VelocityX", "VelocityY", "VelocityZ",
            "AngularY", "Upright", "Grounded",
        ),
    )
    bridge = VrchatOscBridge(config)
    bridge.start()

    if not bridge.thread_alive:
        print("❌ OSC 桥线程未启动，检查端口占用或配置。")
        return 1

    print("✅ OSC 桥已启动。等待 VRChat 回传...\n")
    print("提示：如果一直显示 no_feedback_received，说明 VRChat 没有向 9001 发送任何消息。")
    print("      如果显示 velocity_parameters_absent，说明收到了消息但没有 VelocityX/Y/Z。")
    print("      按 Ctrl+C 退出。\n")

    try:
        while True:
            motion = bridge.motion_feedback(max_age_ms=2000)
            snapshot = bridge.snapshot(include_parameters=False)

            timestamp = time.strftime("%H:%M:%S")
            available = "✅" if motion["available"] else "❌"

            if motion["available"]:
                h_speed = motion["horizontal_speed_mps"]
                v_speed = motion.get("vertical_speed_mps")
                grounded = motion.get("grounded")
                upright = motion.get("upright")
                print(f"[{timestamp}] {available} 水平速度: {h_speed:.3f} m/s  |  "
                      f"垂直: {v_speed:.3f} m/s  |  "
                      f"Grounded: {grounded}  |  Upright: {upright}")
            else:
                reason = motion["reason"]
                expected = motion.get("expected", [])
                present = motion.get("present", [])
                link_age = snapshot.get("link_age_ms", "?")
                print(f"[{timestamp}] {available} reason={reason}  |  "
                      f"链路年龄: {link_age} ms  |  "
                      f"已收到: {len(present)}/{len(expected)}")
                if expected and not present:
                    print(f"            期望参数: {', '.join(expected[:3])}...")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\n停止中...")
        bridge.stop(timeout=1.0)
        print("✅ 已退出。")
        return 0


if __name__ == "__main__":
    sys.exit(main())
