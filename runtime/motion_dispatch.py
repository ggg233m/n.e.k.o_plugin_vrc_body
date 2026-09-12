"""单任务执行调度；世界适配器的创建、运行和释放由同一个所有者管理。"""
import threading

from .motion_execution import MotionExecution


class MotionDispatch:
    def __init__(self, backend, world_factory):
        self.backend = backend
        self.world_factory = world_factory
        self.lock = threading.RLock()
        self.execution = None
        self.world = None
        self.closed = False
        self.stopped = threading.Event()

    def ready(self, session):
        with self.lock:
            return not self.closed and not self.stopped.is_set() and self.world_factory.ready(session)

    def start(self, session, task):
        with self.lock:
            if not self.ready(session):
                raise RuntimeError("execution_bridge_unavailable")
            if self.execution and self.execution.thread.is_alive():
                raise RuntimeError("execution_busy")
            # 成功与失败都必须释放旧适配器；不允许遗留日志线程或 MIDI 句柄。
            if self.world:
                self.world.close()
                self.world = None
            world = self.world_factory(session, task)
            try:
                execution = MotionExecution(self.backend, world)
                self.world, self.execution = world, execution
                if self.closed or self.stopped.is_set():
                    raise RuntimeError("execution_stopped_during_prepare")
                execution.start(task)
            except Exception:
                world.close()
                self.world, self.execution = None, None
                raise

    def prepare(self, session, arguments, instance):
        with self.lock:
            if not self.ready(session):
                raise RuntimeError("execution_bridge_unavailable")
            if self.execution and self.execution.thread.is_alive():
                raise RuntimeError("execution_busy")
            if self.world:
                self.world.close()
                self.world = None
            result = self.world_factory.prepare(session, arguments, instance)
            if self.closed or self.stopped.is_set():
                raise RuntimeError("execution_stopped_during_prepare")
            return result

    def stop(self):
        # 不等待创建锁或网络请求；执行器先打断端口发送。
        self.stopped.set()
        if getattr(self.world_factory, "manages_preparation", False) is True:
            self.world_factory.stop()
        execution = self.execution
        if execution:
            execution.stop()

    def snapshot(self):
        execution = self.execution
        return execution.snapshot() if execution else {"status": "idle"}

    def close(self):
        self.closed = True
        self.stop()
        with self.lock:
            if self.execution:
                self.execution.close()
            if self.world:
                self.world.close()
                self.world = None
            if getattr(self.world_factory, "manages_preparation", False) is True:
                self.world_factory.close()
