"""串行化同一会话的约束和时间轴操作。"""

from functools import wraps


def client_constraints_locked(method):
    """与时间轴回调共用可重入锁，允许批量更新调用单项增删。"""
    @wraps(method)
    def locked(self, client_id, *args, **kwargs):
        session = self.client_sessions.get(client_id)
        if session is None:
            return method(self, client_id, *args, **kwargs)
        with session.constraints_lock:
            return method(self, client_id, *args, **kwargs)

    return locked


def session_constraints_locked(method):
    """生成端读取整批约束时，避免观察到替换过程中的中间状态。"""
    @wraps(method)
    def locked(self, session, *args, **kwargs):
        with session.constraints_lock:
            return method(self, session, *args, **kwargs)

    return locked
