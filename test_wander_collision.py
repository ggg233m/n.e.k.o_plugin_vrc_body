#!/usr/bin/env python
"""直接测试 wander 撞墙行为，绕过 HTTP 和工具层配对检查。"""
import json
import time
from pathlib import Path

# 需要先启动后端进程，这个脚本只是客户端
import requests

BASE = "http://127.0.0.1:14670"
TOKEN = "O6say2hTx5H_-EXrH_7W-N7UQq-eeCKZ"
HEADERS = {"X-Neko-Backend-Token": TOKEN}

def snapshot():
    return requests.get(f"{BASE}/snapshot", headers=HEADERS, timeout=5).json()

def autonomy_goal(text, kind, constraints):
    return requests.post(
        f"{BASE}/autonomy/goal",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"text": text, "kind": kind, "constraints": constraints},
        timeout=5,
    ).json()

def main():
    # 1. 确认 armed
    s = snapshot()
    if not s["autonomy"]["armed"]:
        print("[x] autonomy not armed")
        return

    print(f"[ok] armed, remaining {s['autonomy']['remaining_seconds']:.1f}s")

    # 2. 提交 wander forward，max_duration_s=3.0，turn_deg=0
    print("\n[submit] wander forward (3s)...")
    result = autonomy_goal("walk forward", "wander", {
        "turn_deg": 0,
        "max_duration_s": 3.0,
        "max_forward_axis": 0.5,  # vertical=0.5
    })

    if not result.get("accepted"):
        print(f"[x] rejected: {result.get('reason_code')} - {result.get('reason')}")
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return

    print(f"[ok] accepted")

    # 3. 循环采样，直到 decision_state != 'advance' 且速度归零
    print("\n[sampling]...")
    samples = []
    t0 = time.time()

    while True:
        s = snapshot()
        elapsed = time.time() - t0

        motion = s.get("body_awareness", {}).get("vrchat_osc", {}).get("motion", {})
        hspd = motion.get("speed_mps")

        decision = s.get("autonomy", {}).get("navigation", {}).get("decision", {})
        state = decision.get("state")
        reason = decision.get("reason")

        behavior = s.get("autonomy", {}).get("navigation", {}).get("behavior", {})
        kind = behavior.get("kind")

        sample = {
            "elapsed": elapsed,
            "hspd": hspd,
            "decision_state": state,
            "decision_reason": reason,
            "behavior_kind": kind,
        }
        samples.append(sample)

        print(f"t={elapsed:4.1f}s  hspd={hspd if hspd is not None else 'None':>5}  state={state or 'None':12}  reason={reason or 'None'}")

        # 停止条件：decision_state 不是 advance 且速度为 0 或 None，持续 0.5s
        if state != "advance" and (hspd is None or hspd < 0.01):
            if not hasattr(main, "idle_since"):
                main.idle_since = elapsed
            elif elapsed - main.idle_since > 0.5:
                print("\n[ok] navigator stopped")
                break
        else:
            if hasattr(main, "idle_since"):
                delattr(main, "idle_since")

        if elapsed > 10.0:
            print("\n[timeout]")
            break

        time.sleep(0.1)

    # 4. 保存样本
    out = Path("tmp/wander_collision_direct.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, ensure_ascii=False, indent=1))
    print(f"\n[saved] {len(samples)} samples to {out}")

    # 5. 读取最终的 execution_summary
    final = snapshot()
    summary = final["autonomy"]["navigation"]["behavior"].get("last_outcome", {}).get("execution_summary")
    if summary:
        print("\n=== Execution Summary ===")
        print(json.dumps(summary, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
