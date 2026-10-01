import time
from pathlib import Path

import requests

TERMINAL = ("completed", "failed", "cancelled", "interrupted")


class ModelLabApiError(RuntimeError):
    def __init__(self, status: int, detail):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class ModelLabClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", token: str | None = None, timeout: float = 120):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def _call(self, method: str, path: str, **kwargs):
        response = self.session.request(method, self.base + path, timeout=kwargs.pop("timeout", self.timeout), **kwargs)
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ModelLabApiError(response.status_code, detail)
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response.text

    def get(self, path: str, **params):
        return self._call("GET", path, params={k: v for k, v in params.items() if v is not None})

    def post(self, path: str, json=None, **params):
        return self._call("POST", path, json=json, params={k: v for k, v in params.items() if v is not None})

    def delete(self, path: str):
        return self._call("DELETE", path)

    def health(self):
        return self.get("/health")

    def system(self):
        return self.get("/system")

    def upload_model(self, spec: dict, preprocess: dict | None = None, weights: str | None = None,
                     factory_file: str | None = None, model_id: str | None = None, validate_load: bool = True):
        import json as _json

        handles, files = [], {}
        try:
            for key, value in (("weights", weights), ("factory_file", factory_file)):
                if value:
                    handle = open(value, "rb")
                    handles.append(handle)
                    files[key] = (Path(value).name, handle)
            data = {"spec": _json.dumps(spec), "preprocess": _json.dumps(preprocess or {}),
                    "validate_load": str(validate_load).lower()}
            if model_id:
                data["model_id"] = model_id
            return self._call("POST", "/models", data=data, files=files or None, timeout=3600)
        finally:
            for handle in handles:
                handle.close()

    def register_model(self, spec: dict, preprocess: dict | None = None, model_id: str | None = None,
                       validate_load: bool = True):
        body = {"spec": spec, "preprocess": preprocess or {}, "validate_load": validate_load}
        if model_id:
            body["model_id"] = model_id
        return self.post("/models/register", body)

    def upload_dataset(self, archive: str, dataset_id: str | None = None, labels_csv: str | None = None):
        handles, files = [], {}
        try:
            archive_handle = open(archive, "rb")
            handles.append(archive_handle)
            files["archive"] = (Path(archive).name, archive_handle)
            if labels_csv:
                csv_handle = open(labels_csv, "rb")
                handles.append(csv_handle)
                files["labels_csv"] = (Path(labels_csv).name, csv_handle)
            data = {"dataset_id": dataset_id} if dataset_id else {}
            return self._call("POST", "/datasets", data=data, files=files, timeout=3600)
        finally:
            for handle in handles:
                handle.close()

    def register_dataset(self, path: str, dataset_id: str | None = None, kind: str = "folder",
                         csv_path: str | None = None):
        body = {"path": path, "kind": kind}
        if dataset_id:
            body["dataset_id"] = dataset_id
        if csv_path:
            body["csv_path"] = csv_path
        return self.post("/datasets/register", body)

    def job(self, job_id: str):
        return self.get(f"/jobs/{job_id}")

    def job_log(self, job_id: str, tail: int | None = None):
        return self.get(f"/jobs/{job_id}/log", tail=tail)

    def wait_job(self, job_id: str, timeout: float = 3600, poll: float = 2.0):
        deadline = time.monotonic() + timeout
        while True:
            job = self.job(job_id)
            if job["status"] in TERMINAL or time.monotonic() > deadline:
                return job
            time.sleep(poll)

    def _start(self, path: str, body: dict, wait: bool, timeout: float):
        job = self.post(path, body)
        return self.wait_job(job["job_id"], timeout) if wait else job

    def evaluate(self, wait: bool = False, timeout: float = 3600, **body):
        return self._start("/evaluations", body, wait, timeout)

    def audit(self, wait: bool = False, timeout: float = 3600, **body):
        return self._start("/audits", body, wait, timeout)

    def analyze(self, wait: bool = False, timeout: float = 3600, **body):
        return self._start("/analyses", body, wait, timeout)

    def experiments(self, wait: bool = False, timeout: float = 3600, **body):
        return self._start("/experiments", body, wait, timeout)
