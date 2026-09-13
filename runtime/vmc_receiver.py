"""插件内管理宿主 VMC 接收、T-pose 校准和原设置恢复。"""
from dataclasses import dataclass
import socket
import struct
import threading
import time
import json
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
from .vmc_sharing import Capture

@dataclass(frozen=True)
class VmcConfig:
    enabled: bool = False
    host_api_url: str = 'http://127.0.0.1:48911'
    listen_port: int = 39540

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict): raise ValueError('vmc 必须为配置表')
        enabled=value.get('enabled',False)
        endpoint=value.get('host_api_url',cls.host_api_url)
        port=value.get('listen_port',cls.listen_port)
        if type(enabled) is not bool: raise ValueError('vmc.enabled 必须为布尔值')
        if type(port) is not int or not 1024<=port<=65535: raise ValueError('vmc.listen_port 超出范围')
        if not isinstance(endpoint,str): raise ValueError('vmc.host_api_url 必须为字符串')
        parsed=urlsplit(endpoint)
        if (parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','localhost','::1'}
            or not parsed.port or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {'','/'}): raise ValueError('vmc.host_api_url 必须为带端口的本机 HTTP 地址')
        return cls(enabled,endpoint.rstrip('/'),port)

class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): raise ValueError('VMC API 不允许重定向')

class VmcReceiver:
    def __init__(self,config,*,requester=None):
        self.config=config; self.stop_event=threading.Event(); self.lock=threading.RLock()
        self.thread=None; self.capture=Capture(); self.sample=None; self.state='disabled'
        self.error=None; self.restore_error=None; self.bad_packets=0; self.token=None
        self.requester=requester; self.opener=build_opener(ProxyHandler({}),_NoRedirect())

    def _api(self,path,payload=None):
        if self.requester: return self.requester(path,payload)
        headers={'Origin':self.config.host_api_url,'Accept':'application/json'}
        if payload is not None:
            headers.update({'Content-Type':'application/json','X-CSRF-Token':self.token})
        request=Request(self.config.host_api_url+path,headers=headers,
            data=None if payload is None else json.dumps(payload).encode())
        with self.opener.open(request,timeout=.75) as response:
            raw=response.read(65537)
        if len(raw)>65536: raise ValueError('vmc_response_too_large')
        result=json.loads(raw)
        if not isinstance(result,dict) or result.get('success') is False: raise ValueError('vmc_api_rejected')
        return result

    def latest(self):
        with self.lock:
            sample=self.sample
            if self.stop_event.is_set() or sample is None or not 0<=time.time()-sample['captured_at']<=.5:
                return None
            return sample

    def snapshot(self):
        with self.lock:
            ready=self.latest() is not None
            state=self.state
            if state=='receiving' and not ready: state='source_stale'
            return dict(ready=ready,state=state,error=self.error,restore_error=self.restore_error,
                        frames=self.capture.frames,bad_packets=self.bad_packets)

    def start(self):
        if not self.config.enabled or self.thread: return
        self.thread=threading.Thread(target=self._run,name='yui-vmc-receiver',daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(4)
            if self.thread.is_alive(): raise RuntimeError('VMC 接收器尚未停止，拒绝启动其他动作后端')

    def _run(self):
        from .vmc_sharing import decode_osc
        prior=None; changed=False; sock=None
        target=dict(host='127.0.0.1',port=self.config.listen_port,send_rate_hz=30)
        try:
            self.state='starting'
            sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            if hasattr(socket,'SO_EXCLUSIVEADDRUSE'): sock.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            sock.bind(('127.0.0.1',self.config.listen_port)); sock.settimeout(.05)
            prior=self._api('/api/vmc/status')
            self.token=self._api('/api/config/page_config')['autostart_csrf_token']
            if self.stop_event.is_set(): return
            changed=not prior['enabled'] or any(prior.get(k)!=v for k,v in target.items())
            if changed: self._api('/api/vmc/enable',target)
            if self.stop_event.is_set(): return
            requested=self._api('/api/vmc/t_pose',{'duration_sec':3.})
            generation=requested['t_pose_generation']; deadline=time.monotonic()+12; poll_at=0
            self.state='calibrating'; calibrating=True
            while not self.stop_event.is_set():
                if calibrating and time.monotonic()>=poll_at:
                    status=self._api('/api/vmc/status'); poll_at=time.monotonic()+.2
                    if status.get('t_pose_generation')==generation and not status.get('t_pose_requested'):
                        # 清掉确认之前的 UDP 队列，防止把旧动作当作绑定姿势。
                        sock.setblocking(False)
                        for _ in range(4096):
                            try: sock.recvfrom(65535)
                            except BlockingIOError: break
                        sock.settimeout(.05)
                        self.capture.pending={}; self.capture.started=False; self.capture.calibrate=True
                        calibrating=False; deadline=time.monotonic()+3
                    elif time.monotonic()>deadline:
                        self.state='waiting_model'; return
                try:
                    packet,_=sock.recvfrom(65535)
                    for address,args in decode_osc(packet):
                        sample=self.capture.feed(address,args)
                        with self.lock:
                            if address=='/VMC/Ext/OK' and args==[0]: self.sample=None
                            if sample is not None:
                                self.sample=sample; self.state='receiving'
                except socket.timeout: pass
                except (ValueError,struct.error,UnicodeError,IndexError):
                    self.bad_packets+=1; self.capture.pending={}
                if not calibrating and self.capture.rest is None and time.monotonic()>deadline:
                    self.state='calibration_failed'; return
        except Exception as exc:
            self.state='failed'; self.error=type(exc).__name__
        finally:
            if sock: sock.close()
            with self.lock: self.sample=None
            if prior and changed:
                try:
                    current=self._api('/api/vmc/status')
                    # 用户或其他客户端已经改目的地时，不覆盖新配置。
                    if current.get('enabled') and all(current.get(k)==v for k,v in target.items()):
                        self._api('/api/vmc/enable',{k:prior[k] for k in target})
                        if not prior['enabled']: self._api('/api/vmc/disable',{})
                except Exception as exc: self.restore_error=type(exc).__name__
            if self.stop_event.is_set(): self.state='stopped'
