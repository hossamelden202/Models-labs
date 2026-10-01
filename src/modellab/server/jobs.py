import logging
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from modellab.server.errors import ApiError
from modellab.server.workspace import Workspace, check_id, utcnow

_FORMAT = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
TERMINAL = ("completed", "failed", "cancelled", "interrupted")


class _JobLog(logging.Handler):
    def __init__(self, path: Path, thread_id: int):
        super().__init__(level=logging.INFO)
        self._thread_id = thread_id
        self._file = open(path, "a", encoding="utf-8")
        self.setFormatter(_FORMAT)

    def emit(self, record):
        if record.thread != self._thread_id:
            return
        self._file.write(self.format(record) + "\n")
        self._file.flush()

    def raw(self, text: str) -> None:
        self._file.write(text + "\n")
        self._file.flush()

    def close(self):
        self._file.close()
        super().close()


class JobManager:
    def __init__(self, ws: Workspace, max_workers: int = 1):
        self.ws = ws
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="modellab-job")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        for job in self.list(limit=100000):
            if job["status"] in ("queued", "running"):
                job.update(
                    status="interrupted",
                    finished_at=utcnow(),
                    error={"type": "Interrupted", "message": "the server stopped before the job finished"},
                )
                self._save(job)

    def _save(self, job: dict) -> None:
        self.ws.write_json(self.ws.jobs_dir / f"{job['job_id']}.json", job)

    def _clean(self, text: str) -> str:
        return text.replace(str(self.ws.root), "<workspace>")

    def get(self, job_id: str) -> dict:
        check_id(job_id, "job id")
        path = self.ws.jobs_dir / f"{job_id}.json"
        if not path.is_file():
            raise ApiError(404, f"job '{job_id}' not found")
        return Workspace.read_json(path)

    def list(self, kind: str | None = None, limit: int = 100) -> list[dict]:
        jobs = [Workspace.read_json(p) for p in self.ws.jobs_dir.glob("*.json")]
        if kind:
            jobs = [j for j in jobs if j["kind"] == kind]
        jobs.sort(key=lambda j: j["created_at"], reverse=True)
        return jobs[:limit]

    def log_path(self, job_id: str) -> Path:
        self.get(job_id)
        return self.ws.jobs_dir / f"{job_id}.log"

    def submit(self, kind: str, params: dict, fn) -> dict:
        job = {
            "job_id": uuid.uuid4().hex[:12], "kind": kind, "status": "queued", "params": params,
            "created_at": utcnow(), "started_at": None, "finished_at": None, "result": None, "error": None,
        }
        with self._lock:
            self._save(job)
            self._futures[job["job_id"]] = self._pool.submit(self._run, job["job_id"], fn)
        return job

    def _run(self, job_id: str, fn) -> None:
        with self._lock:
            job = self.get(job_id)
            job.update(status="running", started_at=utcnow())
            self._save(job)
        handler = _JobLog(self.ws.jobs_dir / f"{job_id}.log", threading.get_ident())
        logger = logging.getLogger("modellab")
        logger.addHandler(handler)
        try:
            update = {"status": "completed", "result": fn()}
        except Exception as exc:
            trace = traceback.format_exc()
            handler.raw(trace)
            update = {
                "status": "failed",
                "error": {
                    "type": type(exc).__name__,
                    "message": self._clean(str(exc)),
                    "traceback": self._clean(trace)[-4000:],
                },
            }
        finally:
            logger.removeHandler(handler)
            handler.close()
        with self._lock:
            job = self.get(job_id)
            job.update(update, finished_at=utcnow())
            self._save(job)

    def wait(self, job_id: str, timeout: float) -> dict:
        future = self._futures.get(job_id)
        if future is not None:
            try:
                future.result(timeout)
            except FutureTimeout:
                pass
        return self.get(job_id)

    def cancel(self, job_id: str) -> dict:
        job = self.get(job_id)
        future = self._futures.get(job_id)
        if job["status"] != "queued" or future is None or not future.cancel():
            raise ApiError(409, "only queued jobs can be cancelled")
        with self._lock:
            job.update(status="cancelled", finished_at=utcnow())
            self._save(job)
        return job

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
