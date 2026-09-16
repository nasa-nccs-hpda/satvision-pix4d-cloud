"""Per-process training diagnostics, including failures before Trainer.fit."""
from datetime import datetime, timezone
import faulthandler
import json
import logging
import os
from pathlib import Path
import socket
import sys
import threading
import traceback


def _now():
    return datetime.now(timezone.utc).isoformat()


def _handlers():
    loggers = [logging.getLogger()] + [
        value for value in logging.Logger.manager.loggerDict.values()
        if isinstance(value, logging.Logger)
    ]
    return {handler for logger in loggers for handler in logger.handlers}


class _Tee:
    def __init__(self, stream, log, lock):
        self.stream, self.log, self.lock = stream, log, lock

    def write(self, text):
        with self.lock:
            self.log.write(text)
            self.log.flush()
            return self.stream.write(text)

    def flush(self):
        with self.lock:
            self.log.flush()
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class RunLogging:
    """Tee Python output and logging; preserve exceptions and process identity.

    Separate files prevent distributed ranks/retries overwriting one another.
    Native code writing directly to OS stdout/stderr bypasses Python streams;
    use shell tee as well when investigating native NCCL/driver diagnostics.
    """
    def __init__(self, output_dir, command=None):
        self.directory = Path(output_dir) / 'logs'
        self.command = command if command is not None else sys.argv
        self.exit_code = 0

    def _save_status(self):
        temporary = self.status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.status, indent=2) + '\n')
        temporary.replace(self.status_path)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        local_rank = os.environ.get('LOCAL_RANK', '0')
        node_rank = os.environ.get('NODE_RANK', os.environ.get('GROUP_RANK', '0'))
        rank = os.environ.get('RANK', local_rank if node_rank == '0' else f'node{node_rank}-local{local_rank}')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        stem = f'rank-{rank}-{stamp}-pid{os.getpid()}'
        self.log_path = self.directory / f'{stem}.log'
        self.status_path = self.directory / f'{stem}.status.json'
        self.status = dict(status='running', started_at=_now(), pid=os.getpid(),
                           hostname=socket.gethostname(), rank=rank, local_rank=local_rank,
                           node_rank=node_rank, command=self.command, cwd=os.getcwd(),
                           python=sys.executable, console_log=str(self.log_path.resolve()),
                           cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        self._save_status()
        self.log = self.log_path.open('a', buffering=1)
        self.stdout, self.stderr = sys.stdout, sys.stderr
        lock = threading.RLock()
        sys.stdout = self.tee_stdout = _Tee(self.stdout, self.log, lock)
        sys.stderr = self.tee_stderr = _Tee(self.stderr, self.log, lock)
        # Existing Lightning/DeepSpeed handlers may hold the original streams.
        for handler in _handlers():
            if isinstance(handler, logging.StreamHandler):
                if handler.stream is self.stdout:
                    handler.setStream(self.tee_stdout)
                elif handler.stream is self.stderr:
                    handler.setStream(self.tee_stderr)
        self.root = logging.getLogger()
        self.old_level = self.root.level
        self.root.setLevel(logging.INFO)
        self.handler = None
        if not self.root.handlers:
            self.handler = logging.StreamHandler(self.tee_stderr)
            self.handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
            self.root.addHandler(self.handler)
        self.enabled_fault_handler = not faulthandler.is_enabled()
        if self.enabled_fault_handler:
            faulthandler.enable(file=self.log)
        print(f'[{_now()}] Diagnostics: {self.log_path.resolve()}', flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.status['ended_at'] = _now()
            if exc is None:
                self.status.update(status='completed' if self.exit_code == 0 else 'completed_nonzero',
                                   exit_code=self.exit_code)
            else:
                code = 130 if isinstance(exc, KeyboardInterrupt) else 1
                if isinstance(exc, SystemExit):
                    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
                self.status.update(status=('completed' if code == 0 else
                                           'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'),
                                   exit_code=code, exception=exc_type.__name__, message=str(exc),
                                   traceback=''.join(traceback.format_exception(exc_type, exc, tb)))
                traceback.print_exception(exc_type, exc, tb, file=self.tee_stderr)
            self._save_status()
            print(f'[{_now()}] Run {self.status["status"]}; status: {self.status_path.resolve()}', flush=True)
        finally:
            if self.enabled_fault_handler:
                faulthandler.disable()
            # Include handlers created during the run, so none retain a closed log.
            for handler in _handlers():
                if isinstance(handler, logging.StreamHandler):
                    if handler.stream is self.tee_stdout:
                        handler.setStream(self.stdout)
                    elif handler.stream is self.tee_stderr:
                        handler.setStream(self.stderr)
            if self.handler is not None:
                self.root.removeHandler(self.handler)
                self.handler.close()
            self.root.setLevel(self.old_level)
            sys.stdout, sys.stderr = self.stdout, self.stderr
            self.log.close()
        return False
