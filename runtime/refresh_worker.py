"""宿主集成刷新线程；生命周期 asyncio loop 退出后仍可重试。"""
import threading


class RefreshWorker:
    def __init__(self, callback):
        self.callback = callback
        self.stop_event = threading.Event()
        self.wake = threading.Event()
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        def run():
            delay = .5
            while not self.stop_event.is_set():
                self.wake.wait(delay)
                self.wake.clear()
                if self.stop_event.is_set():
                    break
                try:
                    self.callback()
                    delay = .5
                except Exception:
                    delay = min(10, delay * 2)
        self.thread = threading.Thread(target=run, name="yui-host-refresh", daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.wake.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(2)
