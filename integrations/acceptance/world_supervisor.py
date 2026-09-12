"""使用指定接入包提供临时模型服务，不复用个人目录。"""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time
from urllib.request import urlopen


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--kit-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--retry-marker",type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    kit=args.kit_root.resolve()
    python=kit/".venv/Scripts/python.exe"
    launcher=kit/"portable/launch.py"
    if not python.is_file() or not launcher.is_file():
        parser.error("接入包尚未安装：缺少本机Python环境或启动器")
    config_path=kit/"ardy.local.json"
    if not config_path.is_file(): config_path=kit/"portable/config.example.json"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    port=int(config.get("port",2346))
    startup_timeout=float(config.get("startup_timeout_s",3600))
    if not 1 <= port <= 65535 or not 0 < startup_timeout <= 3600:
        parser.error("接入包端口或启动等待时间无效")
    if port != 2346:
        parser.error("当前世界验收器仅支持端口2346；正常插件连接不受此限制")
    args.output=args.output.resolve()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    log=args.output.with_suffix(".service.log").open("x",encoding="utf-8")
    process=subprocess.Popen([str(python),"-B","-u",str(launcher),"service"],
        cwd=kit,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        deadline=time.monotonic()+startup_timeout
        while time.monotonic()<deadline:
            if process.poll() is not None: raise RuntimeError("owned_service_exited")
            try:
                with urlopen(f"http://127.0.0.1:{port}/health",timeout=.5) as response: health=json.load(response)
                if health.get("ready") is True: break
                if health.get("error"): raise RuntimeError("model_load_failed:"+str(health["error"]))
            except OSError: pass
            time.sleep(.5)
        else: raise RuntimeError("model_start_timeout")
        print("模型已就绪，开始正式协议单客户端验证",flush=True)
        for attempt in range(4):
            output=args.output if attempt==0 else args.output.with_name(args.output.name+"-retry-"+str(attempt))
            result=subprocess.run([sys.executable,"-B","-m",
                "integrations.acceptance.run_world_bridge","--output",str(output)],cwd=root,env=env,timeout=90,
                creationflags=subprocess.CREATE_NO_WINDOW)
            if not result.returncode: return
            if not args.retry_marker or attempt==3: raise SystemExit(result.returncode)
            # 仅收到修复后的显式信号才创建新会话；不自动重放失败任务。
            failed_at=time.time()
            deadline=time.monotonic()+600
            print("实测未通过；服务保留，等待显式重试信号",flush=True)
            while not(args.retry_marker.exists() and args.retry_marker.stat().st_mtime>failed_at):
                if time.monotonic()>deadline or process.poll() is not None: raise RuntimeError("retry_wait_expired")
                time.sleep(.5)
    finally:
        process.terminate()
        try: process.wait(5)
        except subprocess.TimeoutExpired: process.kill();process.wait(5)
        log.close()


if __name__=="__main__": main()
