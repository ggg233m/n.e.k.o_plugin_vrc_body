"""将模型原生调用隔离到自有进程，HTTP租约不依赖推理线程释放GIL。"""
import multiprocessing as mp
import os
import threading


def _worker(connection, options, factory):
    def watch_parent():
        parent=mp.parent_process()
        if parent is not None:
            parent.join()
            # 只退出本工具创建的推理子进程，主服务强退后不能遗留GPU驻留进程。
            os._exit(0)
    threading.Thread(target=watch_parent,daemon=True).start()
    try:
        if factory is None:
            from .ardy_generator import ArdyGenerator
            factory=ArdyGenerator
        model=factory(**options)
        connection.send(('ready',{key:getattr(model,key) for key in
            ('frames','fps','history_limit','load_s','warmup_s','execution_warmup_s')}))
        while True:
            command,args=connection.recv()
            try:
                if command=='candidate':
                    prompt,constraints,history=args
                    if history is not None:
                        history=model.torch.tensor([history],dtype=model.torch.float32,device=model.model.device)
                    pose,sample=model.generate_candidate(prompt,constraints,history)
                    # 只传CPU数值；检查点切片在主服务完成，不共享可被图重放覆盖的GPU存储。
                    result=(pose,sample[0].detach().float().cpu().tolist())
                elif command=='generate':result=model.generate(*args)
                elif command=='reseed':result=model.reseed(*args)
                else:raise ValueError('unknown_generator_command')
                connection.send(('ok',result))
            except Exception as exc:
                connection.send(('error',type(exc).__name__))
    except (EOFError,BrokenPipeError,OSError):
        pass
    except Exception as exc:
        try:connection.send(('error',type(exc).__name__))
        except (BrokenPipeError,OSError):pass
    finally:
        connection.close()


class ProcessGenerator:
    def __init__(self, *, startup_timeout=420., request_timeout=3., factory=None, **options):
        self._ready=False
        self.request_timeout=request_timeout
        self.lock=threading.Lock()
        context=mp.get_context('spawn')
        self.connection,child=context.Pipe()
        self.process=context.Process(target=_worker,args=(child,options,factory),name='ardy-inference',daemon=True)
        self.process.start();child.close()
        try:
            if not self.connection.poll(startup_timeout):raise TimeoutError('generator_startup_timeout')
            kind,metadata=self.connection.recv()
            if kind!='ready':raise RuntimeError('generator_startup_failed:'+str(metadata))
            for key,value in metadata.items():setattr(self,key,value)
            self._ready=True
        except BaseException:
            self.close();raise

    @property
    def ready(self):
        return self._ready and self.process.is_alive()

    def _rpc(self,command,*args):
        with self.lock:
            if not self.ready:raise RuntimeError('generator_process_unavailable')
            try:
                self.connection.send((command,args))
                if not self.connection.poll(self.request_timeout):raise TimeoutError('generator_inference_timeout')
                kind,result=self.connection.recv()
            except (EOFError,OSError,TimeoutError):
                self.close();raise
            if kind!='ok':raise RuntimeError('generator_failed:'+str(result))
            return result

    def generate_candidate(self,prompt,constraints,history=None):
        return self._rpc('candidate',prompt,constraints,history)

    def generate(self,*args):return self._rpc('generate',*args)
    def reseed(self,seed):return self._rpc('reseed',seed)

    def history_at(self,sample,generated_frames):
        if (type(generated_frames) is not int or not 0<=generated_frames<=self.frames
            or generated_frames%4 or not self.frames<=len(sample)<=self.frames+self.history_limit):
            raise ValueError('invalid_history_boundary')
        end=len(sample)-self.frames+generated_frames
        return [row[:] for row in sample[max(0,end-self.history_limit):end]] if end else None

    def close(self):
        self._ready=False
        if self.process.is_alive():
            self.process.terminate();self.process.join(3)
            if self.process.is_alive():self.process.kill();self.process.join(3)
        self.connection.close()
