"""直接测试 LocalNavigator 的撞墙检测，跳过 service/autonomy 层。"""
import sys
import time
import json
from pathlib import Path

# 模拟最小的依赖环境
class FakeVrchatOscBridge:
    def __init__(self):
        self.packets = []

    def locomotion(self, vertical, horizontal, duration_ms):
        self.packets.append({
            "t": time.time(),
            "vertical": vertical,
            "horizontal": horizontal,
            "duration_ms": duration_ms,
        })
        return True

    def turn(self, horizontal, duration_ms):
        self.packets.append({
            "t": time.time(),
            "turn": horizontal,
            "duration_ms": duration_ms,
        })
        return True

class FakeDirectionMemory:
    def should_refuse(self, bearing_deg, reason):
        return None  # 不拒绝任何方向

    def record_progress(self, bearing_deg, cleared, blocked, progress_m):
        pass

# 导入 navigator（作为包导入避免相对导入错误）
sys.path.insert(0, str(Path(__file__).parent))
from backend.navigator import LocalNavigator

def main():
    bridge = FakeVrchatOscBridge()
    direction_memory = FakeDirectionMemory()

    nav = LocalNavigator(
        bridge=bridge,
        direction_memory=direction_memory,
        tick_rate=10.0,
        capacity_ticks=240,
    )

    # 模拟一个简单的 wander forward 目标
    print("[raise_goal] wander forward, max_duration_s=5.0")
    nav.raise_goal(
        kind="wander",
        text="walk forward into wall",
        target_id=None,
        selector=None,
        constraints={
            "turn_deg": 0,
            "max_duration_s": 5.0,
            "max_forward_axis": 0.5,
        },
    )

    # 手动驱动 tick 循环，每 100ms 喂一次 motion 反馈
    print("\n[tick loop] 开始...")
    samples = []
    t0 = time.time()

    for i in range(100):  # 最多 10 秒
        elapsed = time.time() - t0

        # 模拟 motion 反馈：前 2 秒正常移动，之后速度降到接近零（撞墙）
        if elapsed < 2.0:
            motion = {
                "available": True,
                "speed_mps": 4.0,  # 正常巡航速度
                "grounded": True,
                "value_age_ms": 50,
            }
        else:
            motion = {
                "available": True,
                "speed_mps": 0.08,  # 顶墙速度
                "grounded": True,
                "value_age_ms": 50,
            }

        decision = nav.tick(motion)

        samples.append({
            "t": elapsed,
            "state": decision.state,
            "reason": decision.reason,
            "speed_mps": motion["speed_mps"],
        })

        print(f"t={elapsed:5.2f}s  state={decision.state:8s}  reason={decision.reason:25s}  speed={motion['speed_mps']:.2f}")

        if decision.state is None:
            print("\n[done] navigator 已停止")
            break

        time.sleep(0.1)

    # 保存样本
    out = Path("tmp/navigator_direct_test.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[saved] {len(samples)} samples to {out}")

    # 检查发出的 OSC 包
    print(f"\n[osc packets] {len(bridge.packets)} sent")
    for p in bridge.packets[:5]:
        print(f"  {p}")
    if len(bridge.packets) > 5:
        print(f"  ... and {len(bridge.packets) - 5} more")

if __name__ == "__main__":
    main()
