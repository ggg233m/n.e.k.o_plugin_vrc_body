"""通过 obs-websocket 控制 OBS 录制。

用途：让录制流程全自动 —— 开始录制、跑走位脚本、停止录制，不需要手点。

OBS 侧前提：工具 → WebSocket 服务器设置 → 启用 WebSocket 服务器。
连接信息（IP/端口/密码）在 OBS 的「WebSocket 连接信息」窗口里。

用法：
    python obs_ctl.py status            # 看当前录制状态与视频设置
    python obs_ctl.py start             # 开始录制
    python obs_ctl.py stop              # 停止录制
    python obs_ctl.py fix-video         # 设为 1080p60（录制规格要求）
"""

import argparse
import base64
import hashlib
import json
import os
import sys

import websocket          # websocket-client


def load_obs_config():
    """从 OBS 自己的配置读 WebSocket 端口与密码。

    比手抄「服务器密码」可靠：那串随机字符里的 l/L、0/O 在截图里根本分不出来
    （实际踩过：``Mon1itLy`` 的大写 L 被读成小写 l，认证直接失败）。
    """
    path = os.path.join(
        os.path.expandvars(r"%APPDATA%\obs-studio\plugin_config\obs-websocket"),
        "config.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("server_password") or "", int(data.get("server_port") or 4455)
    except Exception:
        return "", 4455


class ObsError(RuntimeError):
    pass


class Obs:
    def __init__(self, host: str, port: int, password: str, timeout: float = 10.0,
                 debug: bool = False):
        url = "ws://%s:%d" % (host, port)
        if debug:
            print("[debug] 连接 %s" % url)
        self.ws = websocket.create_connection(url, timeout=timeout)
        if debug:
            print("[debug] 握手完成，subprotocol=%s" % self.ws.getsubprotocol())
        raw = self.ws.recv()
        if debug:
            print("[debug] 原始 Hello: %r" % (raw[:300] if raw else raw))
        if not raw:
            raise ObsError("连接建立但未收到 Hello（OBS 是否只允许单客户端？）")
        try:
            hello = json.loads(raw)
        except Exception as exc:
            raise ObsError("Hello 解析失败：%s；原始=%r" % (exc, raw[:200]))
        if hello.get("op") != 0:
            raise ObsError("期望 Hello(op=0)，实际 op=%s" % hello.get("op"))
        ident = {"op": 1, "d": {"rpcVersion": 1}}
        auth = hello.get("d", {}).get("authentication")
        if auth:
            if not password:
                raise ObsError("OBS 要求密码，但未提供 --password")
            secret = base64.b64encode(
                hashlib.sha256((password + auth["salt"]).encode()).digest()
            ).decode()
            ident["d"]["authentication"] = base64.b64encode(
                hashlib.sha256((secret + auth["challenge"]).encode()).digest()
            ).decode()
        self.ws.send(json.dumps(ident))
        reply = json.loads(self.ws.recv())
        if reply.get("op") != 2:
            raise ObsError("认证失败：%s" % json.dumps(reply, ensure_ascii=False)[:200])
        self._seq = 0

    def request(self, request_type: str, **kwargs) -> dict:
        self._seq += 1
        rid = str(self._seq)
        payload = {"op": 6, "d": {"requestType": request_type,
                                  "requestId": rid}}
        payload["d"].update(kwargs)
        self.ws.send(json.dumps(payload))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("op") != 7:
                continue                       # 跳过事件
            d = msg["d"]
            if d.get("requestId") != rid:
                continue
            status = d.get("requestStatus", {})
            if not status.get("result"):
                raise ObsError("%s 失败：%s" % (request_type,
                                              status.get("comment", "unknown")))
            return d.get("responseData", {}) or {}

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


def show_status(obs: Obs) -> dict:
    video = obs.request("GetVideoSettings")
    try:
        rec = obs.request("GetRecordStatus")
    except ObsError:
        rec = {}
    fps = None
    if video.get("fpsDenominator"):
        fps = video["fpsNumerator"] / video["fpsDenominator"]
    print("分辨率   %sx%s（输出 %sx%s）"
          % (video.get("baseWidth"), video.get("baseHeight"),
             video.get("outputWidth"), video.get("outputHeight")))
    print("帧率     %s" % ("%.0f fps" % fps if fps else "未知"))
    print("录制中   %s%s" % (rec.get("outputActive"),
                           "" if not rec.get("outputActive")
                           else "  已录 %.1f s" % (rec.get("outputDuration", 0) / 1000.0)))
    return {"video": video, "recording": rec, "fps": fps}


def fix_video(obs: Obs) -> None:
    """把 OBS 设成录制规格要求的 1080p60。

    帧率是刚需（决定帧间旋转量），分辨率不是——所以这里只保证 1080p60，
    码率/编码器仍由 OBS 自己的输出设置决定，可在 设置>输出 里调。
    """
    cur = obs.request("GetVideoSettings")
    print("当前：%sx%s @ %s fps"
          % (cur.get("baseWidth"), cur.get("baseHeight"),
             "%.0f" % (cur["fpsNumerator"] / cur["fpsDenominator"])
             if cur.get("fpsDenominator") else "?"))
    obs.request("SetVideoSettings",
                baseWidth=1920, baseHeight=1080,
                outputWidth=1920, outputHeight=1080,
                fpsNumerator=60, fpsDenominator=1)
    after = obs.request("GetVideoSettings")
    print("已设为：%sx%s @ %s fps"
          % (after.get("baseWidth"), after.get("baseHeight"),
             "%.0f" % (after["fpsNumerator"] / after["fpsDenominator"])
             if after.get("fpsDenominator") else "?"))
    print("提示：编码器与码率仍需在 OBS「设置 > 输出」里确认（建议 H.264 / 约 20 Mbps）")


def main() -> int:
    ap = argparse.ArgumentParser(description="OBS 录制控制（obs-websocket）")
    ap.add_argument("action", choices=["status", "start", "stop", "fix-video"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None,
                    help="默认从 OBS 配置读取（通常 4455）")
    ap.add_argument("--password", default=None,
                    help="默认从 OBS 配置读取，避免手抄出错")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    if a.port is None or a.password is None:
        auto_pw, auto_port = load_obs_config()
        if a.port is None:
            a.port = auto_port
        if a.password is None:
            a.password = auto_pw
            if a.password:
                print("已从 OBS 配置读取密码（%d 字符）" % len(a.password))
    if not a.password:
        print("未能读取密码，请用 --password 指定")

    obs = None
    try:
        obs = Obs(a.host, a.port, a.password, debug=a.debug)
    except Exception as exc:
        print("无法连接 OBS WebSocket（%s:%d）：%s" % (a.host, a.port, exc))
        print("请确认 OBS 已启用 WebSocket 服务器，且密码正确。")
        return 2

    try:
        if a.action == "status":
            show_status(obs)
        elif a.action == "fix-video":
            fix_video(obs)
        elif a.action == "start":
            show_status(obs)
            obs.request("StartRecord")
            print("→ 已开始录制")
        elif a.action == "stop":
            data = obs.request("StopRecord")
            out = data.get("outputPath")
            print("→ 已停止录制%s" % ("  文件：%s" % out if out else ""))
    except ObsError as exc:
        print("操作失败：%s" % exc)
        return 1
    finally:
        obs.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
