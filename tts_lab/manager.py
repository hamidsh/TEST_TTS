import multiprocessing as mp
import queue
import threading
import time
import uuid
from dataclasses import asdict
from .config import ROOT, save_config
from .worker import run


class BusyError(RuntimeError):
    pass


class Manager:
    """One model, one worker, one active job. All public operations are serialized."""
    def __init__(self):
        self.lock = threading.RLock()
        self.ctx = mp.get_context('spawn')
        self.process = self.commands = self.events = None
        self.jobs = {}
        self.active = None
        self.loaded = None
        self.details = None

    def _drain(self):
        if self.events:
            while True:
                try:
                    event = self.events.get_nowait()
                except queue.Empty:
                    break
                job = self.jobs.get(event['id'])
                if not job:
                    continue
                kind = event['kind']
                if kind == 'progress':
                    job['message'] = event['message']
                elif kind == 'chunk':
                    job['chunks'].append(event['url'])
                elif kind in ('done', 'error'):
                    job['state'] = kind
                    job['result'] = event.get('result')
                    job['message'] = event.get('message', 'آماده')
                    self.active = None
                    if job['action'] == 'load':
                        self.loaded = job['config'] if kind == 'done' else None
                        self.details = event.get('result')
        if self.process and not self.process.is_alive():
            self.loaded = None
            self.details = None
            if self.active:
                job = self.jobs[self.active]
                job.update(state='error', message=f'Worker stopped (exit {self.process.exitcode}). See logs/worker.log; check RAM/VRAM.')
                self.active = None

    def _stop(self):
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=2)
            self.process.close()
        for q in (self.commands, self.events):
            if q:
                q.close()
                q.cancel_join_thread()
        self.process = self.commands = self.events = None
        self.loaded = self.details = None

    def stop(self):
        with self.lock:
            self._drain()
            if self.active:
                self.jobs[self.active].update(state='cancelled', message='متوقف شد؛ مدل از حافظه خارج شد.')
                self.active = None
            self._stop()

    def _start(self):
        self.commands = self.ctx.Queue()
        self.events = self.ctx.Queue()
        self.process = self.ctx.Process(target=run, args=(self.commands, self.events, str(ROOT)), daemon=True)
        self.process.start()

    def submit(self, action, **payload):
        with self.lock:
            self._drain()
            if self.active:
                raise BusyError('یک عملیات در حال اجراست؛ صبر کنید یا توقف را بزنید.')
            if action == 'load':
                cfg = payload.pop('config')
                cfg.validate()  # validate before unloading an existing model
                if self.loaded == asdict(cfg):
                    return {'already_loaded': True}
                self._stop()
                self._start()
                payload['config'] = asdict(cfg)
                save_config(cfg)
            elif action == 'generate':
                if not self.loaded:
                    raise ValueError('ابتدا مدل را بارگذاری کنید.')
                payload['config'] = self.loaded
            else:
                raise ValueError('Unknown action')
            jid = uuid.uuid4().hex
            self.jobs[jid] = dict(id=jid, action=action, state='running', message='شروع عملیات…',
                                  chunks=[], created=time.time(), config=payload['config'])
            self.active = jid
            # Bound in-memory history; audio/metrics stay on disk.
            while len(self.jobs) > 100:
                del self.jobs[next(iter(self.jobs))]
            self.commands.put(dict(id=jid, action=action, **payload))
            return {'id': jid}

    def snapshot(self):
        with self.lock:
            self._drain()
            return {'loaded': self.loaded, 'details': self.details, 'active': self.active,
                    'jobs': list(self.jobs.values())[-10:],
                    'worker_pid': self.process.pid if self.process and self.process.is_alive() else None}
