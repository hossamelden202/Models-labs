import requests


class AiApiError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


def _message(resp):
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:300]
    if isinstance(body, dict):
        for key in ("detail", "error", "message"):
            if key in body:
                return str(body[key])[:300]
    return str(body)[:300]


class AiClient:
    def __init__(self, base_url, session=None, timeout=600):
        self.base = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update({"ngrok-skip-browser-warning": "true"})
        self.timeout = timeout

    def _call(self, method, path, **kwargs):
        resp = getattr(self.session, method)(self.base + path, timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            raise AiApiError(resp.status_code, _message(resp))
        return resp.json()

    def research(self, family_id, question):
        return self._call("post", "/ai/research", json={"family_id": family_id, "question": question})

    def list_research(self):
        return self._call("get", "/ai/research")

    def get(self, research_id):
        return self._call("get", f"/ai/research/{research_id}")

    def approve(self, research_id, model_id=None, dataset_id=None, device="auto", force=False, include_control=None):
        body = {"device": device, "force": force}
        if include_control is not None:
            body["include_control"] = include_control
        if model_id:
            body["model_id"] = model_id
        if dataset_id:
            body["dataset_id"] = dataset_id
        return self._call("post", f"/ai/research/{research_id}/approve", json=body)

    def reject(self, research_id):
        return self._call("post", f"/ai/research/{research_id}/reject")
