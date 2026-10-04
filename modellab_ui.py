# ============================================================
# MODELLAB — LOCAL STREAMLIT UI
# Phases 1–5
#
# LOCAL PC UI
# Backend: FastAPI running on Kaggle
# ============================================================

from __future__ import annotations

import json
import tempfile
import time
from typing import Any

import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ModelLab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_API_URL = "http://127.0.0.1:8000"

DEFAULT_DATASET_PATH = "/path/to/dataset"
DEFAULT_MODEL_PATH = "/path/to/model.pt"
DEFAULT_FACTORY_CALLABLE = ""


# ============================================================
# SESSION STATE
# ============================================================

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL

if "api_token" not in st.session_state:
    st.session_state.api_token = ""

if "connected" not in st.session_state:
    st.session_state.connected = False

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None


# ============================================================
# HELPERS
# ============================================================

def normalize_url(url: str) -> str:
    url = url.strip()

    if not url:
        return DEFAULT_API_URL

    return url.rstrip("/")


def pretty_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        return str(value)


def extract_data(response: Any) -> Any:
    if not isinstance(response, dict):
        return response

    if "data" in response:
        return response["data"]

    return response


def extract_items(response: Any, preferred_keys: tuple[str, ...]) -> list:
    """
    Handles common API response shapes:

        {"items": [...]}
        {"models": [...]}
        {"data": [...]}
        {"data": {"models": [...]}}
        [...]

    """
    if response is None:
        return []

    if isinstance(response, list):
        return response

    if not isinstance(response, dict):
        return []

    for key in preferred_keys:
        value = response.get(key)

        if isinstance(value, list):
            return value

    data = response.get("data")

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in preferred_keys:
            value = data.get(key)

            if isinstance(value, list):
                return value

    return []


def get_model_id(item: dict) -> str | None:
    return (
        item.get("model_id")
        or item.get("id")
        or item.get("name")
    )


def get_dataset_id(item: dict) -> str | None:
    return (
        item.get("dataset_id")
        or item.get("id")
        or item.get("name")
    )


def show_response(response: Any) -> None:
    with st.expander("Backend response", expanded=False):
        st.json(response)


def response_ok(response: Any) -> bool:
    if not isinstance(response, dict):
        return False

    if response.get("ok") is True:
        return True

    status = response.get("status")

    if isinstance(status, str):
        return status.lower() in {
            "ok",
            "success",
            "completed",
        }

    return False


def get_job_id(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None

    candidates = [
        response.get("job_id"),
        response.get("id"),
    ]

    data = response.get("data")

    if isinstance(data, dict):
        candidates.extend(
            [
                data.get("job_id"),
                data.get("id"),
            ]
        )

    for candidate in candidates:
        if candidate:
            return str(candidate)

    return None


def parse_csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_numeric_list(value: str) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def temp_file_from_upload(uploaded_file: Any, suffix: str = "") -> str | None:
    if uploaded_file is None:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getvalue())
        return handle.name


def make_model_spec(
    *,
    model_id: str,
    source: str,
    checkpoint_path: str,
    factory_callable: str,
    class_names: str,
    num_classes: int,
    input_width: int,
    input_height: int,
    state_dict_key: str = "model_state_dict",
    architecture: str | None = None,
    hidden_size: int | None = None,
    patch_size: int | None = None,
    backbone_path: str | None = None,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "model_id": model_id,
        "source": source,
        "num_classes": int(num_classes),
        "class_names": parse_csv_values(class_names),
        "input_size": [int(input_width), int(input_height)],
        "state_dict_key": state_dict_key,
    }

    if checkpoint_path.strip():
        spec["path"] = checkpoint_path.strip()

    if factory_callable.strip():
        spec["factory"] = factory_callable.strip()

    if architecture:
        spec["architecture"] = architecture
    if hidden_size is not None:
        spec["hidden_size"] = int(hidden_size)
    if patch_size is not None:
        spec["patch_size"] = int(patch_size)
    if backbone_path:
        spec["backbone_path"] = backbone_path
    if mean:
        spec["mean"] = mean
    if std:
        spec["std"] = std

    return spec


def _normalize_hint_list(value: Any) -> list[str] | None:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return items
    return None


def artifact_resolution_defaults(
    *,
    artifact_id: str,
    inspection: dict[str, Any] | None = None,
    preferred_architecture: str | None = None,
) -> dict[str, Any]:
    """Build the backend resolution payload from artifact inspection results.

    The UI should never force a handwritten PyTorch state_dict spec for a
    TensorFlow or self-contained artifact. The artifact inspection result is the
    source of truth, and only missing facts should be filled in manually.
    """
    payload: dict[str, Any] = {"artifact_id": artifact_id}
    info = inspection or {}
    fmt = str(info.get("format") or "unknown").lower()
    hints = info.get("hints") or []

    if not isinstance(hints, list):
        hints = [hints]

    hint_map: dict[str, Any] = {}
    for hint in hints:
        if not isinstance(hint, dict):
            continue
        field = hint.get("field")
        if field:
            hint_map[field] = hint.get("value")

    class_names = hint_map.get("class_names")
    if isinstance(class_names, list) and all(isinstance(v, str) for v in class_names):
        payload["class_names"] = list(class_names)

    num_classes = hint_map.get("num_classes")
    if num_classes is not None:
        payload["num_classes"] = int(num_classes)

    input_size = hint_map.get("input_size")
    if input_size is not None:
        payload["input_size"] = [int(value) for value in input_size]

    if isinstance(hint_map.get("preprocess.mean"), list):
        payload.setdefault("preprocess", {})["mean"] = [float(v) for v in hint_map["preprocess.mean"]]

    if isinstance(hint_map.get("preprocess.std"), list):
        payload.setdefault("preprocess", {})["std"] = [float(v) for v in hint_map["preprocess.std"]]

    if hint_map.get("architecture_hint"):
        payload["architecture"] = str(hint_map["architecture_hint"])
    elif preferred_architecture:
        payload["architecture"] = str(preferred_architecture)

    state_dicts = info.get("state_dicts") or []
    if fmt in {"torch_archive", "safetensors"} and state_dicts:
        first_key = state_dicts[0].get("key_path")
        if first_key:
            payload["state_dict_key"] = first_key

    if fmt == "pickle_object":
        payload["trust_executable"] = False

    if fmt in {"tensorflow_savedmodel", "keras"}:
        payload.pop("state_dict_key", None)
        payload.pop("architecture", None)

    return payload


def make_analysis_config(
    *,
    min_support: int,
    alpha: float,
    high_confidence_threshold: float,
    low_confidence_threshold: float,
    max_evidence: int,
    max_listed_slices: int,
    top_confusions: int,
    min_risk_difference: float,
) -> dict[str, Any]:
    return {
        "min_support": int(min_support),
        "alpha": float(alpha),
        "high_confidence_threshold": float(high_confidence_threshold),
        "low_confidence_threshold": float(low_confidence_threshold),
        "max_evidence": int(max_evidence),
        "max_listed_slices": int(max_listed_slices),
        "top_confusions": int(top_confusions),
        "min_risk_difference": float(min_risk_difference),
    }


def make_inference_spec(*, name: str, claim: str, intervention_kind: str, threshold: float, fallback_class: str, analysis_id: str | None = None) -> dict[str, Any]:
    intervention: dict[str, Any] = {"kind": intervention_kind}
    if intervention_kind == "inference":
        intervention["confidence_threshold"] = float(threshold)
        intervention["fallback_class"] = fallback_class

    spec: dict[str, Any] = {
        "name": name,
        "hypothesis": {
            "claim": claim,
            "expected_direction": "improve",
        },
        "intervention": intervention,
        "metrics": ["accuracy", "macro_f1"],
        "seed": 42,
        "repetitions": 1,
        "n_bootstrap": 1000,
        "n_permutations": 5000,
        "alpha": 0.05,
        "practical_threshold": 0.01,
    }

    if analysis_id:
        spec["population"] = {"analysis_id": analysis_id}

    return spec


# ============================================================
# AUDIT RESULT HELPERS
# ============================================================

def extract_audit_id(response: Any) -> str | None:
    """Extract an audit ID from an audit response or completed job."""
    if not isinstance(response, dict):
        return None

    candidates = [
        response.get("audit_id"),
    ]

    result = response.get("result")
    if isinstance(result, dict):
        candidates.append(result.get("audit_id"))

    data = response.get("data")
    if isinstance(data, dict):
        candidates.append(data.get("audit_id"))
        nested_result = data.get("result")
        if isinstance(nested_result, dict):
            candidates.append(nested_result.get("audit_id"))

    for value in candidates:
        if value:
            return str(value)

    return None


def audit_status_label(status: str) -> str:
    labels = {
        "queued": "⏳ Queued",
        "running": "🔄 Running",
        "completed": "✅ Completed",
        "failed": "❌ Failed",
        "cancelled": "⚪ Cancelled",
        "interrupted": "⚠️ Interrupted",
    }
    return labels.get(str(status).lower(), str(status).title())


def render_audit_finding(finding: dict, index: int) -> None:
    """Render one Finding from the Phase 3 AuditReport."""
    if not isinstance(finding, dict):
        st.write(finding)
        return

    severity = str(finding.get("severity", "info")).lower()
    icons = {
        "high": "🔴",
        "medium": "🟠",
        "low": "🟡",
        "info": "ℹ️",
    }
    icon = icons.get(severity, "•")

    title = finding.get("title") or finding.get("code") or "Untitled finding"
    category = finding.get("category", "unknown")
    count = finding.get("count", 0)

    label = f"{icon} {str(severity).upper()} · {title}"
    with st.expander(label, expanded=(severity in {"high", "medium"})):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.caption("Category")
            st.write(str(category))

        with col2:
            st.caption("Affected count")
            st.write(str(count))

        with col3:
            st.caption("Finding code")
            st.write(str(finding.get("code", "—")))

        description = finding.get("description")
        if description:
            st.markdown("**What was found**")
            st.write(str(description))

        affected_classes = finding.get("affected_classes")
        if affected_classes:
            st.markdown("**Affected classes**")
            st.write(affected_classes)

        affected_splits = finding.get("affected_splits")
        if affected_splits:
            st.markdown("**Affected splits**")
            st.write(affected_splits)

        sample_ids = finding.get("sample_ids")
        if sample_ids:
            st.markdown("**Example sample IDs**")
            st.write(sample_ids)

        details = finding.get("details")
        if details:
            st.markdown("**Evidence / details**")
            st.json(details)

        if finding.get("evidence_truncated"):
            st.caption(
                "Evidence was truncated according to the audit reporting limits."
            )


def render_audit_report(report: dict) -> None:
    """Render a complete AuditReport returned by the backend."""
    if not isinstance(report, dict):
        st.error("The backend returned an invalid audit report.")
        st.json(report)
        return

    # Some API wrappers may return {"data": {...}}.
    report = extract_data(report)
    if not isinstance(report, dict):
        st.error("The backend returned an invalid audit report.")
        st.json(report)
        return

    audit_id = report.get("audit_id", "—")
    fingerprint = report.get("fingerprint", "—")
    scan = report.get("scan") or {}
    summary = report.get("summary") or {}
    findings = report.get("findings") or []

    st.markdown("## Dataset Audit Report")

    top1, top2, top3, top4 = st.columns(4)

    with top1:
        st.metric("Samples", str(
            report.get("num_samples")
            or scan.get("num_samples")
            or scan.get("total")
            or "—"
        ))

    with top2:
        st.metric("Total findings", str(len(findings)))

    with top3:
        st.metric("Audit ID", str(audit_id))

    with top4:
        st.metric("Fingerprint", str(fingerprint))

    # Prefer the authoritative report findings to recompute counts.
    severity_counts = {
        "high": 0,
        "medium": 0,
        "low": 0,
        "info": 0,
    }

    for finding in findings:
        if isinstance(finding, dict):
            severity = str(finding.get("severity", "info")).lower()
            if severity in severity_counts:
                severity_counts[severity] += 1

    # If the backend has no findings list but has summary counts, preserve them.
    if not findings:
        reported = summary.get("findings_by_severity")
        if isinstance(reported, dict):
            for key in severity_counts:
                severity_counts[key] = int(reported.get(key, 0) or 0)

    st.markdown("### Findings Summary")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("🔴 High", severity_counts["high"])

    with c2:
        st.metric("🟠 Medium", severity_counts["medium"])

    with c3:
        st.metric("🟡 Low", severity_counts["low"])

    with c4:
        st.metric("ℹ️ Info", severity_counts["info"])

    if not findings:
        st.success("No detailed findings were returned by the audit report.")
        return

    st.divider()

    severity_order = ("high", "medium", "low", "info")
    severity_titles = {
        "high": "🔴 High severity",
        "medium": "🟠 Medium severity",
        "low": "🟡 Low severity",
        "info": "ℹ️ Informational findings",
    }

    for severity in severity_order:
        group = [
            finding
            for finding in findings
            if isinstance(finding, dict)
            and str(finding.get("severity", "info")).lower() == severity
        ]

        if not group:
            continue

        st.markdown(f"### {severity_titles[severity]} ({len(group)})")

        for index, finding in enumerate(group):
            render_audit_finding(finding, index)


# ============================================================
# API CLIENT
# ============================================================

class ModelLabAPI:
    def __init__(self, base_url: str, token: str = ""):
        self.base_url = normalize_url(base_url)
        self.token = token.strip()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
        }

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict | None = None,
        files: dict | None = None,
        data: dict | None = None,
        timeout: int = 120,
    ) -> dict:
        url = f"{self.base_url}{path}"

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self._headers(),
                json=json_body,
                params=params,
                files=files,
                data=data,
                timeout=timeout,
            )

        except requests.RequestException as exc:
            return {
                "ok": False,
                "error": str(exc),
                "url": url,
            }

        try:
            payload = response.json()
        except Exception:
            payload = {
                "text": response.text,
            }

        if isinstance(payload, dict):
            payload.setdefault(
                "http_status",
                response.status_code,
            )

            if response.ok:
                payload.setdefault("ok", True)
            else:
                payload.setdefault("ok", False)

            return payload

        return {
            "ok": response.ok,
            "http_status": response.status_code,
            "data": payload,
        }

    # --------------------------------------------------------
    # Health / system
    # --------------------------------------------------------

    def health(self) -> dict:
        return self._request(
            "GET",
            "/health",
            timeout=15,
        )

    def system(self) -> dict:
        return self._request(
            "GET",
            "/system",
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    def models(self) -> dict:
        return self._request(
            "GET",
            "/models",
        )

    def register_model(
        self,
        model_id: str,
        spec: dict,
        preprocess: dict,
        validate_load: bool = True,
    ) -> dict:
        return self._request(
            "POST",
            "/models/register",
            json_body={
                "model_id": model_id,
                "spec": spec,
                "preprocess": preprocess,
                "validate_load": validate_load,
            },
            timeout=300,
        )

    def upload_model(
        self,
        spec: dict,
        preprocess: dict | None = None,
        weights: str | None = None,
        factory_file: str | None = None,
        model_id: str | None = None,
        validate_load: bool = True,
    ) -> dict:
        files: dict[str, Any] = {}
        handles: list[Any] = []

        try:
            for key, path in [("weights", weights), ("factory_file", factory_file)]:
                if not path:
                    continue
                handle = open(path, "rb")
                handles.append(handle)
                files[key] = (path.split("/")[-1], handle)

            data = {
                "spec": json.dumps(spec),
                "preprocess": json.dumps(preprocess or {}),
                "validate_load": str(bool(validate_load)).lower(),
            }
            if model_id:
                data["model_id"] = model_id

            return self._request(
                "POST",
                "/models",
                data=data,
                files=files,
                timeout=600,
            )
        finally:
            for handle in handles:
                handle.close()

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    def datasets(self) -> dict:
        return self._request(
            "GET",
            "/datasets",
        )

    def register_dataset(
        self,
        dataset_id: str,
        path: str,
        kind: str = "folder",
        csv_path: str | None = None,
        path_column: str = "path",
        label_column: str = "label",
    ) -> dict:
        return self._request(
            "POST",
            "/datasets/register",
            json_body={
                "dataset_id": dataset_id,
                "path": path,
                "kind": kind,
                "csv_path": csv_path,
                "path_column": path_column,
                "label_column": label_column,
            },
            timeout=120,
        )

    def upload_dataset(
        self,
        archive: str,
        dataset_id: str | None = None,
        labels_csv: str | None = None,
        path_column: str = "path",
        label_column: str = "label",
    ) -> dict:
        files: dict[str, Any] = {}
        handles: list[Any] = []

        try:
            archive_handle = open(archive, "rb")
            handles.append(archive_handle)
            files["archive"] = (archive.split("/")[-1], archive_handle)

            if labels_csv:
                csv_handle = open(labels_csv, "rb")
                handles.append(csv_handle)
                files["labels_csv"] = (labels_csv.split("/")[-1], csv_handle)

            data = {
                "dataset_id": dataset_id or "",
                "path_column": path_column,
                "label_column": label_column,
            }
            return self._request(
                "POST",
                "/datasets",
                data={k: v for k, v in data.items() if v not in (None, "")},
                files=files,
                timeout=600,
            )
        finally:
            for handle in handles:
                handle.close()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def evaluate(
        self,
        model_id: str,
        dataset_id: str,
        subpath: str | None = None,
        evaluation_id: str | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        device: str = "auto",
        seed: int = 42,
    ) -> dict:
        return self._request(
            "POST",
            "/evaluations",
            json_body={
                "model_id": model_id,
                "dataset_id": dataset_id,
                "subpath": subpath,
                "evaluation_id": evaluation_id,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "device": device,
                "seed": seed,
            },
            timeout=300,
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    def audit(
        self,
        dataset_id: str,
        subpath: str | None = None,
        audit_id: str | None = None,
        config: dict | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/audits",
            json_body={
                "dataset_id": dataset_id,
                "subpath": subpath,
                "audit_id": audit_id,
                "config": config or {},
            },
            timeout=300,
        )

    def audit_report(self, audit_id: str) -> dict:
        return self._request(
            "GET",
            f"/audits/{audit_id}",
            timeout=120,
        )

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    def analyze(
        self,
        evaluation_id: str,
        audit_id: str | None = None,
        analysis_id: str | None = None,
        config: dict | None = None,
    ) -> dict:
        return self._request(
            "POST",
            "/analyses",
            json_body={
                "evaluation_id": evaluation_id,
                "audit_id": audit_id,
                "analysis_id": analysis_id,
                "config": config or {},
            },
            timeout=300,
        )

    # --------------------------------------------------------
    # Experiments
    # --------------------------------------------------------

    def experiments(
        self,
        family_id: str,
        baseline_evaluation_id: str | None = None,
        specs: list[dict] | None = None,
        matrices: list[dict] | None = None,
        model_id: str | None = None,
        dataset_id: str | None = None,
        subpath: str | None = None,
        device: str = "auto",
        force: bool = False,
        stop_on_failure: bool = False,
        alpha: float = 0.05,
    ) -> dict:
        return self._request(
            "POST",
            "/experiments",
            json_body={
                "family_id": family_id,
                "baseline_evaluation_id": baseline_evaluation_id,
                "specs": specs or [],
                "matrices": matrices or [],
                "model_id": model_id,
                "dataset_id": dataset_id,
                "subpath": subpath,
                "device": device,
                "force": force,
                "stop_on_failure": stop_on_failure,
                "alpha": alpha,
            },
            timeout=300,
        )

    # --------------------------------------------------------
    # Jobs
    # --------------------------------------------------------

    def jobs(self) -> dict:
        return self._request(
            "GET",
            "/jobs",
        )

    def job(self, job_id: str) -> dict:
        return self._request(
            "GET",
            f"/jobs/{job_id}",
        )

    def job_log(self, job_id: str) -> dict:
        return self._request(
            "GET",
            f"/jobs/{job_id}/log",
        )


# ============================================================
# API INSTANCE
# ============================================================

api = ModelLabAPI(
    st.session_state.api_url,
    st.session_state.api_token,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧪 ModelLab")

    st.caption("AI Model Laboratory")

    st.divider()

    st.markdown("### Backend")

    api_url = st.text_input(
        "FastAPI URL",
        value=st.session_state.api_url,
        key="api_url_input",
        help="URL of the ModelLab FastAPI backend.",
    )

    token = st.text_input(
        "API token",
        value=st.session_state.api_token,
        type="password",
        key="api_token_input",
    )

    if st.button(
        "Connect",
        use_container_width=True,
        type="primary",
    ):
        st.session_state.api_url = normalize_url(api_url)
        st.session_state.api_token = token.strip()

        api = ModelLabAPI(
            st.session_state.api_url,
            st.session_state.api_token,
        )

        result = api.health()

        if result.get("ok"):
            st.session_state.connected = True
            st.success("Connected")
        else:
            st.session_state.connected = False
            st.error("Connection failed")

            with st.expander("Details"):
                st.json(result)

    if st.session_state.connected:
        st.success("● Backend connected")
    else:
        st.warning("● Not connected")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Models",
            "Datasets",
            "Laboratory",
            "Results",
            "Failure Explorer",
            "Experiments",
            "Jobs",
        ],
    )

    st.divider()

    st.caption(
        "ModelLab\n"
        "Deterministic ML investigation platform"
    )


# ============================================================
# REFRESH API INSTANCE
# ============================================================

api = ModelLabAPI(
    st.session_state.api_url,
    st.session_state.api_token,
)


# ============================================================
# CONNECTION CHECK
# ============================================================

if not st.session_state.connected:
    st.title("🧪 ModelLab")
    st.subheader("Connect to the Kaggle backend")

    st.info(
        "Enter the public FastAPI/ngrok URL from your Kaggle backend "
        "in the sidebar, then click Connect."
    )

    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("🧪 ModelLab Dashboard")

    st.caption(
        "Experimental investigation environment for ML model failures."
    )

    health = api.health()
    system = api.system()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Backend",
            "Online" if health.get("ok") else "Offline",
        )

    with col2:
        models_response = api.models()
        models = extract_items(
            models_response,
            ("models", "items"),
        )
        st.metric(
            "Models",
            len(models),
        )

    with col3:
        datasets_response = api.datasets()
        datasets = extract_items(
            datasets_response,
            ("datasets", "items"),
        )
        st.metric(
            "Datasets",
            len(datasets),
        )

    with col4:
        jobs_response = api.jobs()
        jobs = extract_items(
            jobs_response,
            ("jobs", "items"),
        )
        st.metric(
            "Jobs",
            len(jobs),
        )

    st.divider()

    st.markdown("## Investigation Pipeline")

    pipeline = [
        ("1", "Foundation", "Contracts and infrastructure"),
        ("2", "Evaluation", "Baseline model evaluation"),
        ("3", "Audit", "Dataset integrity investigation"),
        ("4", "Failure Mining", "Failure slices and patterns"),
        ("5", "Experiments", "Controlled interventions"),
        ("6", "Hypotheses", "Competing explanations"),
        ("7", "Investigation", "Evidence-backed mechanism"),
        ("8", "Repair", "Targeted model/data repair"),
        ("9", "API + UI", "Experiment laboratory interface"),
        ("10", "Validation", "Regression and final benchmark"),
    ]

    cols = st.columns(5)

    for index, (number, name, description) in enumerate(pipeline):

        with cols[index % 5]:
            st.markdown(
                f"""
                <div style="
                    border:1px solid rgba(128,128,128,0.35);
                    border-radius:10px;
                    padding:14px;
                    margin-bottom:12px;
                    min-height:130px;
                ">
                    <div style="font-size:26px;font-weight:700;">
                        {number}
                    </div>
                    <div style="font-size:17px;font-weight:600;">
                        {name}
                    </div>
                    <div style="font-size:13px;opacity:0.75;">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    st.markdown("### System")

    if system:
        show_response(system)


# ============================================================
# MODELS
# ============================================================

elif page == "Models":

    st.title("🧠 Models")

    models_response = api.models()

    if not models_response.get("ok"):
        st.error("Could not retrieve models.")
        show_response(models_response)
        st.stop()

    models = extract_items(
        models_response,
        ("models", "items"),
    )

    if models:
        st.markdown("### Registered Models")

        for model in models:
            model_id = get_model_id(model) or "unknown"
            with st.expander(
                f"🧠 {model_id}",
                expanded=False,
            ):
                st.json(model)
    else:
        st.info("No models registered yet.")

    st.divider()

    st.markdown("### Register a model")
    st.caption(
        "Choose the artifact, inspect it with the backend, and fill in only the missing model metadata. "
        "The UI follows the backend's actual capability instead of assuming every checkpoint is a PyTorch state_dict."
    )

    with st.form("register_model_form"):
        model_id = st.text_input("Model ID", value="demo_model")
        source_kind = st.radio(
            "Source",
            ["Upload", "Server path", "Existing artifact"],
            horizontal=True,
        )

        if source_kind == "Existing artifact":
            artifact_id = st.text_input(
                "Artifact ID",
                value="",
                help="Use an already staged artifact from the backend.",
            )
            local_path = ""
            artifact_file = None
        elif source_kind == "Server path":
            artifact_id = ""
            local_path = st.text_input(
                "Server path",
                value="",
                help="Local path on the backend machine or Kaggle input directory.",
            )
            artifact_file = None
        else:
            artifact_id = ""
            local_path = ""
            artifact_file = st.file_uploader(
                "Artifact file",
                type=["pt", "pth", "ckpt", "bin", "safetensors", "onnx", "pb", "keras", "py"],
                help="Upload an artifact to inspect before registration.",
            )

        factory_file = st.file_uploader("Factory file (optional)", type=["py"], help="Optional Python file when the model is built through a custom factory.")
        architecture_hint = st.text_input("Architecture hint", value="", help="Optional. The backend can infer it from the artifact when available.")
        class_names_text = st.text_input("Class names", value="", help="Only fill this in if the backend did not infer classes from the artifact.")
        num_classes = st.number_input("Number of classes", min_value=1, max_value=1000, value=3, step=1)
        input_width = st.number_input("Input width", min_value=1, max_value=4096, value=224, step=1)
        input_height = st.number_input("Input height", min_value=1, max_value=4096, value=224, step=1)
        state_dict_key = st.text_input("State dict key", value="", help="Only used for nested or multi-state checkpoints.")
        trust_executable = st.checkbox("I trust executable model code", value=False)
        validate_load = st.checkbox("Validate model load", value=True)

        submit_model = st.form_submit_button("Inspect + Register Model", type="primary", use_container_width=True)

    if submit_model:
        try:
            files: dict[str, Any] = {}
            handles: list[Any] = []
            try:
                if artifact_file is not None:
                    weight_path = temp_file_from_upload(artifact_file, ".bin")
                    if weight_path:
                        handle = open(weight_path, "rb")
                        handles.append(handle)
                        files["artifact"] = (artifact_file.name or "artifact.bin", handle)
                if factory_file is not None:
                    factory_path = temp_file_from_upload(factory_file, ".py")
                    if factory_path:
                        handle = open(factory_path, "rb")
                        handles.append(handle)
                        files["factory_file"] = (factory_file.name or "factory.py", handle)

                stage_data: dict[str, str] = {}
                if source_kind == "Existing artifact" and artifact_id.strip():
                    stage_data["artifact_id"] = artifact_id.strip()
                elif source_kind == "Server path" and local_path.strip():
                    stage_data["local_path"] = local_path.strip()

                inspection_response = api._request(
                    "POST",
                    "/model-artifacts",
                    data=stage_data,
                    files=files,
                    timeout=300,
                )

                if not inspection_response.get("ok"):
                    st.error("Artifact inspection failed.")
                    show_response(inspection_response)
                    st.stop()

                payload = artifact_resolution_defaults(
                    artifact_id=inspection_response.get("artifact_id") or artifact_id.strip() or local_path.strip() or "uploaded_model",
                    inspection=inspection_response.get("inspection") or {},
                    preferred_architecture=architecture_hint.strip() or None,
                )

                if class_names_text.strip():
                    payload["class_names"] = parse_csv_values(class_names_text)

                payload["num_classes"] = int(num_classes)
                payload["input_size"] = [int(input_width), int(input_height)]

                if state_dict_key.strip():
                    payload["state_dict_key"] = state_dict_key.strip()
                if trust_executable:
                    payload["trust_executable"] = True

                resolution = api._request(
                    "POST",
                    "/models/resolve",
                    json_body=payload,
                    timeout=300,
                )

                st.session_state.last_response = resolution

                if resolution.get("status") == "ready":
                    registration = api._request(
                        "POST",
                        "/models/from-artifact",
                        json_body={**payload, "model_id": model_id.strip() or (payload.get("artifact_id") or "demo_model")},
                        timeout=300,
                    )
                    st.session_state.last_response = registration
                    if registration.get("ok"):
                        st.success(f"Model `{model_id}` registered successfully.")
                        show_response(registration)
                    else:
                        st.error("Registration failed.")
                        show_response(registration)
                else:
                    st.info("The backend needs a little more information before the model can be registered.")
                    st.json({"payload": payload, "resolution": resolution})
                    show_response(resolution)
            finally:
                for handle in handles:
                    try:
                        handle.close()
                    except Exception:
                        pass
        except Exception as exc:
            st.error(f"Invalid model configuration: {exc}")


# ============================================================
# DATASETS
# ============================================================

elif page == "Datasets":

    st.title("🗂️ Datasets")

    datasets_response = api.datasets()

    if not datasets_response.get("ok"):
        st.error("Could not retrieve datasets.")
        show_response(datasets_response)
        st.stop()

    datasets = extract_items(
        datasets_response,
        ("datasets", "items"),
    )

    if datasets:
        st.markdown("### Registered Datasets")

        for dataset in datasets:

            dataset_id = get_dataset_id(dataset) or "unknown"

            with st.expander(
                f"🗂️ {dataset_id}",
                expanded=False,
            ):
                st.json(dataset)

    else:
        st.info("No datasets registered yet.")

    st.divider()

    st.markdown("### Register a dataset")

    with st.form("register_dataset_form"):
        dataset_id = st.text_input("Dataset ID", value="demo_dataset")
        dataset_mode = st.radio(
            "Dataset registration mode",
            ["Register server folder", "Upload dataset archive"],
            horizontal=True,
        )

        dataset_path = st.text_input(
            "Dataset path on server",
            value="/path/to/dataset",
            help="Local folder already present on the backend server.",
        )
        archive_file = st.file_uploader(
            "Dataset archive (zip)",
            type=["zip"],
            help="Upload a zip archive with the dataset folder inside it.",
        )

        dataset_kind = st.selectbox("Dataset type", ["folder", "csv"], index=0)
        uploaded_labels_csv = st.file_uploader(
            "Labels CSV file (optional)",
            type=["csv"],
            help="Optional CSV labels file used by the backend for CSV datasets.",
        )
        csv_path = st.text_input("CSV path", value="", help="Used for CSV datasets on the server.")
        path_column = st.text_input("Path column", value="path")
        label_column = st.text_input("Label column", value="label")

        submitted = st.form_submit_button("Register Dataset", type="primary", use_container_width=True)

    if submitted:
        try:
            with st.spinner("Registering dataset..."):
                if dataset_mode == "Upload dataset archive":
                    archive_path = temp_file_from_upload(archive_file, ".zip")
                    csv_path_temp = temp_file_from_upload(uploaded_labels_csv, ".csv")
                    try:
                        result = api.upload_dataset(
                            archive=archive_path or "",
                            dataset_id=dataset_id,
                            labels_csv=csv_path_temp,
                            path_column=path_column,
                            label_column=label_column,
                        )
                    finally:
                        for path in [archive_path, csv_path_temp]:
                            if path:
                                try:
                                    import os
                                    os.unlink(path)
                                except FileNotFoundError:
                                    pass
                else:
                    result = api.register_dataset(
                        dataset_id=dataset_id,
                        path=dataset_path,
                        kind=dataset_kind,
                        csv_path=csv_path.strip() or None,
                        path_column=path_column,
                        label_column=label_column,
                    )

            st.session_state.last_response = result
            if result.get("ok"):
                st.success(f"Dataset `{dataset_id}` registered successfully.")
            else:
                st.error("Dataset registration failed.")
            show_response(result)
        except Exception as exc:
            st.error(f"Invalid dataset configuration: {exc}")


# ============================================================
# LABORATORY
# ============================================================

elif page == "Laboratory":

    st.title("🔬 Laboratory")

    operation = st.selectbox(
        "Laboratory operation",
        [
            "Phase 2 — Baseline Evaluation",
            "Phase 3 — Dataset Audit",
            "Phase 4 — Failure Analysis",
            "Phase 5 — Controlled Experiments",
        ],
    )

    # ========================================================
    # PHASE 2 — EVALUATION
    # ========================================================

    if operation == "Phase 2 — Baseline Evaluation":

        st.subheader("Baseline Evaluation")

        models_response = api.models()
        datasets_response = api.datasets()

        models = extract_items(
            models_response,
            ("models", "items"),
        )

        datasets = extract_items(
            datasets_response,
            ("datasets", "items"),
        )

        if not models:
            st.warning("Register a model first.")
            st.stop()

        if not datasets:
            st.warning("Register a dataset first.")
            st.stop()

        model_map = {
            str(get_model_id(item)): item
            for item in models
            if get_model_id(item)
        }

        dataset_map = {
            str(get_dataset_id(item)): item
            for item in datasets
            if get_dataset_id(item)
        }

        selected_model = st.selectbox(
            "Model",
            list(model_map.keys()),
        )

        selected_dataset = st.selectbox(
            "Dataset",
            list(dataset_map.keys()),
        )

        subpath = st.text_input(
            "Subpath",
            value="",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            batch_size = st.number_input(
                "Batch size",
                min_value=1,
                max_value=2048,
                value=32,
                step=1,
            )

        with col2:
            num_workers = st.number_input(
                "Workers",
                min_value=0,
                max_value=64,
                value=0,
                step=1,
            )

        with col3:
            device = st.selectbox(
                "Device",
                [
                    "auto",
                    "cpu",
                    "cuda",
                ],
            )

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2**32 - 1,
            value=42,
            step=1,
        )

        evaluation_id = st.text_input(
            "Evaluation ID",
            value="",
        )

        if st.button(
            "▶ RUN BASELINE EVALUATION",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner("Running baseline evaluation..."):

                result = api.evaluate(
                    model_id=selected_model,
                    dataset_id=selected_dataset,
                    subpath=subpath.strip() or None,
                    evaluation_id=evaluation_id.strip() or None,
                    batch_size=int(batch_size),
                    num_workers=int(num_workers),
                    device=device,
                    seed=int(seed),
                )

            st.session_state.last_response = result

            if result.get("ok"):
                st.success("Evaluation submitted successfully.")
            else:
                st.error("Evaluation failed.")

            show_response(result)

    # ========================================================
    # PHASE 3 — DATASET AUDIT
    # ========================================================

    elif operation == "Phase 3 — Dataset Audit":

        st.subheader("Dataset Audit")

        st.caption(
            "ModelLab scans the dataset for label integrity, image "
            "integrity, duplicates, leakage indicators, distribution "
            "problems, statistics, and outliers."
        )

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        datasets_response = api.datasets()

        if not datasets_response.get("ok"):
            st.error("Could not retrieve datasets.")
            show_response(datasets_response)
            st.stop()

        datasets = extract_items(
            datasets_response,
            ("datasets", "items"),
        )

        if not datasets:
            st.warning(
                "No registered datasets found. "
                "Register a dataset first."
            )
            st.stop()

        dataset_map = {}

        for item in datasets:

            if not isinstance(item, dict):
                continue

            dataset_id = get_dataset_id(item)

            if dataset_id:
                dataset_map[str(dataset_id)] = item

        if not dataset_map:
            st.error(
                "The backend returned datasets, but no dataset IDs "
                "were found."
            )

            show_response(datasets_response)

            st.stop()

        selected_dataset = st.selectbox(
            "Dataset",
            list(dataset_map.keys()),
            key="audit_dataset",
        )

        dataset_info = dataset_map[selected_dataset]

        with st.expander(
            "Dataset information",
            expanded=False,
        ):
            st.json(dataset_info)

        subpath = st.text_input(
            "Subpath",
            value="",
            key="audit_subpath",
            help=(
                "Optional path inside the registered dataset."
            ),
        )

        # ----------------------------------------------------
        # DATASET STRUCTURE
        # ----------------------------------------------------

        st.markdown("### Dataset Structure")

        layout = st.radio(
            "Dataset layout",
            options=[
                "auto",
                "class_only",
                "split_class",
            ],
            index=0,
            horizontal=True,
            key="audit_layout",
            help=(
                "auto: detect automatically. "
                "class_only: class directories contain images. "
                "split_class: split directories contain class "
                "directories."
            ),
        )

        st.caption(
            "Auto is recommended unless you specifically need to "
            "force a layout."
        )

        # ----------------------------------------------------
        # LABEL / CSV SETTINGS
        # ----------------------------------------------------

        with st.expander(
            "Label / CSV settings",
            expanded=False,
        ):

            class_names_text = st.text_input(
                "Expected class names",
                value="",
                key="audit_class_names",
                help=(
                    "Optional comma-separated class names. "
                    "Leave empty to infer classes."
                ),
            )

            path_column = st.text_input(
                "Path column",
                value="path",
                key="audit_path_column",
            )

            label_column = st.text_input(
                "Label column",
                value="label",
                key="audit_label_column",
            )

            label_name_column = st.text_input(
                "Label-name column",
                value="",
                key="audit_label_name_column",
                help=(
                    "Optional CSV column containing human-readable "
                    "class names."
                ),
            )

            split_column = st.text_input(
                "Split column",
                value="split",
                key="audit_split_column",
                help=(
                    "Leave empty if there is no split column."
                ),
            )

            id_column = st.text_input(
                "Sample ID column",
                value="",
                key="audit_id_column",
                help=(
                    "Optional CSV column containing stable sample IDs."
                ),
            )

        # ----------------------------------------------------
        # CHECKS
        # ----------------------------------------------------

        st.markdown("### Checks Performed")

        st.info(
            "Phase 3 runs these checks automatically. "
            "The controls below are informational only; they are "
            "not sent as unsupported boolean configuration fields."
        )

        checks_col1, checks_col2 = st.columns(2)

        with checks_col1:

            st.checkbox(
                "Class / label integrity",
                value=True,
                disabled=True,
                key="audit_check_labels",
            )

            st.checkbox(
                "Image integrity",
                value=True,
                disabled=True,
                key="audit_check_images",
            )

            st.checkbox(
                "Exact duplicates",
                value=True,
                disabled=True,
                key="audit_check_exact_duplicates",
            )

            st.checkbox(
                "Near duplicates",
                value=True,
                disabled=True,
                key="audit_check_near_duplicates",
            )

        with checks_col2:

            st.checkbox(
                "Split / leakage indicators",
                value=True,
                disabled=True,
                key="audit_check_leakage",
            )

            st.checkbox(
                "Class distribution",
                value=True,
                disabled=True,
                key="audit_check_distribution",
            )

            st.checkbox(
                "Image statistics",
                value=True,
                disabled=True,
                key="audit_check_statistics",
            )

            st.checkbox(
                "Outlier detection",
                value=True,
                disabled=True,
                key="audit_check_outliers",
            )

        # ----------------------------------------------------
        # ADVANCED SETTINGS
        # ----------------------------------------------------

        with st.expander(
            "Advanced audit settings",
            expanded=False,
        ):

            st.markdown("#### Duplicate Detection")

            dup_col1, dup_col2 = st.columns(2)

            with dup_col1:

                near_duplicate_threshold = st.number_input(
                    "Near-duplicate threshold",
                    min_value=0,
                    max_value=16,
                    value=4,
                    step=1,
                    key="audit_near_duplicate_threshold",
                    help=(
                        "Maximum dHash Hamming distance for "
                        "near-duplicate matching."
                    ),
                )

            with dup_col2:

                max_bucket_size = st.number_input(
                    "Maximum hash bucket size",
                    min_value=2,
                    max_value=100000,
                    value=5000,
                    step=100,
                    key="audit_max_bucket_size",
                )

            st.markdown("#### Image Integrity")

            img_col1, img_col2 = st.columns(2)

            with img_col1:

                min_side = st.number_input(
                    "Minimum image side",
                    min_value=1,
                    max_value=10000,
                    value=32,
                    step=1,
                    key="audit_min_side",
                )

                min_file_bytes = st.number_input(
                    "Minimum file size (bytes)",
                    min_value=0,
                    max_value=1024 * 1024 * 1024,
                    value=256,
                    step=64,
                    key="audit_min_file_bytes",
                )

            with img_col2:

                max_side = st.number_input(
                    "Maximum image side",
                    min_value=1,
                    max_value=10000,
                    value=10000,
                    step=100,
                    key="audit_max_side",
                )

                max_file_bytes = st.number_input(
                    "Maximum file size (bytes)",
                    min_value=1,
                    max_value=10 * 1024 * 1024 * 1024,
                    value=50 * 1024 * 1024,
                    step=1024 * 1024,
                    key="audit_max_file_bytes",
                )

            max_aspect_ratio = st.number_input(
                "Maximum aspect ratio",
                min_value=1.01,
                max_value=100.0,
                value=4.0,
                step=0.1,
                key="audit_max_aspect_ratio",
            )

            st.markdown("#### Distribution")

            dist_col1, dist_col2 = st.columns(2)

            with dist_col1:

                imbalance_ratio = st.number_input(
                    "Imbalance warning ratio",
                    min_value=1.01,
                    max_value=1000.0,
                    value=5.0,
                    step=0.5,
                    key="audit_imbalance_ratio",
                )

                split_proportion_gap = st.number_input(
                    "Split proportion gap",
                    min_value=0.001,
                    max_value=1.0,
                    value=0.15,
                    step=0.01,
                    key="audit_split_proportion_gap",
                )

            with dist_col2:

                severe_imbalance_ratio = st.number_input(
                    "Severe imbalance ratio",
                    min_value=1.01,
                    max_value=1000.0,
                    value=20.0,
                    step=0.5,
                    key="audit_severe_imbalance_ratio",
                )

                min_split_size_for_gap = st.number_input(
                    "Minimum split size",
                    min_value=1,
                    max_value=1000000,
                    value=20,
                    step=1,
                    key="audit_min_split_size_for_gap",
                )

            st.markdown("#### Statistics / Outliers")

            stat_col1, stat_col2 = st.columns(2)

            with stat_col1:

                compute_stats = st.checkbox(
                    "Compute image statistics",
                    value=True,
                    key="audit_compute_stats",
                    help=(
                        "Compute brightness, contrast, and "
                        "saturation statistics."
                    ),
                )

                outlier_z = st.number_input(
                    "Outlier robust-z threshold",
                    min_value=0.01,
                    max_value=100.0,
                    value=4.0,
                    step=0.5,
                    key="audit_outlier_z",
                )

            with stat_col2:

                min_samples_for_outliers = st.number_input(
                    "Minimum samples for outliers",
                    min_value=3,
                    max_value=1000000,
                    value=20,
                    step=1,
                    key="audit_min_samples_for_outliers",
                )

                min_contrast = st.number_input(
                    "Minimum contrast",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.01,
                    step=0.01,
                    key="audit_min_contrast",
                )

            st.markdown("#### Evidence / Reporting")

            evidence_col1, evidence_col2 = st.columns(2)

            with evidence_col1:

                filename_overlap_min_stem = st.number_input(
                    "Minimum filename stem length",
                    min_value=1,
                    max_value=1000,
                    value=6,
                    step=1,
                    key="audit_filename_overlap_min_stem",
                )

                max_evidence = st.number_input(
                    "Maximum evidence samples",
                    min_value=1,
                    max_value=10000,
                    value=20,
                    step=1,
                    key="audit_max_evidence",
                )

            with evidence_col2:

                max_listed_groups = st.number_input(
                    "Maximum listed groups",
                    min_value=1,
                    max_value=100000,
                    value=200,
                    step=10,
                    key="audit_max_listed_groups",
                )

        # ----------------------------------------------------
        # AUDIT ID
        # ----------------------------------------------------

        audit_id = st.text_input(
            "Audit ID",
            value="",
            key="audit_id",
            help=(
                "Optional. Leave empty for automatic ID generation."
            ),
        )

        # ----------------------------------------------------
        # BUILD ACTUAL AuditConfig
        # ----------------------------------------------------

        class_names = None

        if class_names_text.strip():

            class_names = tuple(
                value.strip()
                for value in class_names_text.split(",")
                if value.strip()
            )

        audit_config = {
            "layout": layout,
            "class_names": (
                list(class_names)
                if class_names
                else None
            ),

            "path_column": (
                path_column.strip()
                or "path"
            ),

            "label_column": (
                label_column.strip()
                or "label"
            ),

            "label_name_column": (
                label_name_column.strip()
                or None
            ),

            "split_column": (
                split_column.strip()
                or None
            ),

            "id_column": (
                id_column.strip()
                or None
            ),

            "near_duplicate_threshold": int(
                near_duplicate_threshold
            ),

            "max_bucket_size": int(
                max_bucket_size
            ),

            "min_side": int(
                min_side
            ),

            "max_side": int(
                max_side
            ),

            "max_aspect_ratio": float(
                max_aspect_ratio
            ),

            "min_file_bytes": int(
                min_file_bytes
            ),

            "max_file_bytes": int(
                max_file_bytes
            ),

            "imbalance_ratio": float(
                imbalance_ratio
            ),

            "severe_imbalance_ratio": float(
                severe_imbalance_ratio
            ),

            "split_proportion_gap": float(
                split_proportion_gap
            ),

            "min_split_size_for_gap": int(
                min_split_size_for_gap
            ),

            "outlier_z": float(
                outlier_z
            ),

            "min_samples_for_outliers": int(
                min_samples_for_outliers
            ),

            "compute_stats": bool(
                compute_stats
            ),

            "min_contrast": float(
                min_contrast
            ),

            "filename_overlap_min_stem": int(
                filename_overlap_min_stem
            ),

            "max_evidence": int(
                max_evidence
            ),

            "max_listed_groups": int(
                max_listed_groups
            ),
        }

        # ----------------------------------------------------
        # CLIENT-SIDE VALIDATION
        # ----------------------------------------------------

        config_valid = True

        if severe_imbalance_ratio < imbalance_ratio:

            st.error(
                "Severe imbalance ratio must not be below "
                "the normal imbalance ratio."
            )

            config_valid = False

        if min_side > max_side:

            st.error(
                "Minimum image side cannot exceed maximum "
                "image side."
            )

            config_valid = False

        if min_file_bytes > max_file_bytes:

            st.error(
                "Minimum file size cannot exceed maximum "
                "file size."
            )

            config_valid = False

        # ----------------------------------------------------
        # CONFIG PREVIEW
        # ----------------------------------------------------

        with st.expander(
            "Audit configuration preview",
            expanded=False,
        ):
            st.json(audit_config)

        # ----------------------------------------------------
        # RUN AUDIT
        # ----------------------------------------------------

        st.divider()

        run_audit = st.button(
            "▶ RUN AUDIT",
            type="primary",
            use_container_width=True,
            disabled=not config_valid,
            key="run_phase3_audit",
        )

        if run_audit:

            with st.spinner(
                "Submitting Phase 3 dataset audit..."
            ):

                result = api.audit(
                    dataset_id=selected_dataset,
                    subpath=(
                        subpath.strip()
                        or None
                    ),
                    audit_id=(
                        audit_id.strip()
                        or None
                    ),
                    config=audit_config,
                )

            st.session_state.last_response = result

            if not result.get("ok"):
                st.error("Dataset audit submission failed.")
                show_response(result)

            else:
                job_id = get_job_id(result)
                returned_audit_id = extract_audit_id(result)

                # The audit endpoint is asynchronous. Follow the job until
                # completion so the UI reports the actual audit result.
                if job_id:
                    progress = st.empty()
                    status_box = st.empty()

                    max_wait_seconds = 300
                    poll_interval = 2
                    elapsed = 0

                    while elapsed < max_wait_seconds:
                        job_response = api.job(str(job_id))
                        job_data = extract_data(job_response)

                        if isinstance(job_data, dict):
                            status = str(
                                job_data.get("status", "unknown")
                            ).lower()

                            progress.progress(
                                min(
                                    elapsed / max_wait_seconds,
                                    0.99,
                                )
                            )

                            status_box.info(
                                f"Audit job `{job_id}` — "
                                f"{audit_status_label(status)}"
                            )

                            if status == "completed":
                                returned_audit_id = (
                                    extract_audit_id(job_data)
                                    or returned_audit_id
                                )
                                progress.progress(1.0)
                                status_box.success(
                                    "✅ Dataset audit completed."
                                )
                                break

                            if status in {
                                "failed",
                                "cancelled",
                                "interrupted",
                            }:
                                status_box.error(
                                    f"Dataset audit "
                                    f"{audit_status_label(status)}."
                                )
                                show_response(job_response)
                                break

                        time.sleep(poll_interval)
                        elapsed += poll_interval

                    else:
                        status_box.warning(
                            "The audit is still running. "
                            "You can monitor it from the Jobs page."
                        )

                if returned_audit_id:
                    st.info(
                        f"Audit ID: `{returned_audit_id}`"
                    )

                    with st.spinner(
                        "Loading detailed audit findings..."
                    ):
                        report_response = api.audit_report(
                            returned_audit_id
                        )

                    if report_response.get("ok"):
                        render_audit_report(report_response)
                    else:
                        st.warning(
                            "The audit completed, but the detailed "
                            "audit report could not be retrieved."
                        )
                        show_response(report_response)

                elif not job_id:
                    # Backward-compatible handling for a synchronous audit
                    # response that directly contains the report.
                    data = extract_data(result)

                    if isinstance(data, dict) and (
                        data.get("findings")
                        or data.get("summary")
                        or data.get("audit_id")
                    ):
                        render_audit_report(data)
                    else:
                        st.success(
                            "Dataset audit completed, but no audit ID "
                            "was returned."
                        )

                show_response(result)

    # ========================================================
    # PHASE 4 — FAILURE ANALYSIS
    # ========================================================

    elif operation == "Phase 4 — Failure Analysis":

        st.subheader("Failure Mining & Analysis")

        st.caption(
            "Analyze an existing evaluation and optionally connect "
            "it to a dataset audit."
        )

        evaluation_id = st.text_input(
            "Evaluation ID",
            value="",
            help="ID returned by Phase 2 evaluation.",
        )

        audit_id = st.text_input(
            "Audit ID",
            value="",
            help="Optional Phase 3 audit ID.",
        )

        analysis_id = st.text_input(
            "Analysis ID",
            value="",
        )

        st.markdown("### Analysis configuration")

        analysis_config = {
            "high_confidence_threshold": st.slider("High confidence threshold", 0.5, 1.0, 0.9, 0.01),
            "low_confidence_threshold": st.slider("Low confidence threshold", 0.1, 1.0, 0.6, 0.01),
            "min_support": st.number_input("Minimum support", min_value=2, max_value=1000, value=20, step=1),
            "alpha": st.slider("Significance alpha", 0.01, 0.2, 0.05, 0.01),
            "max_evidence": st.number_input("Max evidence", min_value=1, max_value=200, value=20, step=1),
            "max_listed_slices": st.number_input("Max listed slices", min_value=1, max_value=500, value=100, step=1),
            "top_confusions": st.number_input("Top confusions", min_value=1, max_value=50, value=5, step=1),
            "min_risk_difference": st.slider("Min risk difference", 0.0, 0.5, 0.05, 0.01),
        }

        analysis_config = make_analysis_config(**analysis_config)

        if st.button(
            "▶ RUN FAILURE ANALYSIS",
            type="primary",
            use_container_width=True,
            disabled=(
                not evaluation_id.strip()
                or analysis_config is None
            ),
        ):

            with st.spinner(
                "Running failure analysis..."
            ):

                result = api.analyze(
                    evaluation_id=evaluation_id.strip(),
                    audit_id=(
                        audit_id.strip()
                        or None
                    ),
                    analysis_id=(
                        analysis_id.strip()
                        or None
                    ),
                    config=analysis_config,
                )

            st.session_state.last_response = result

            if result.get("ok"):

                st.success(
                    "Failure analysis submitted successfully."
                )

            else:

                st.error(
                    "Failure analysis failed."
                )

            show_response(result)


    # ========================================================
    # PHASE 5 — CONTROLLED EXPERIMENTS
    # ========================================================

    elif operation == "Phase 5 — Controlled Experiments":

        st.subheader(
            "Controlled Experiments"
        )

        st.caption(
            "Run a controlled experiment family against a baseline."
        )

        family_id = st.text_input(
            "Experiment family ID",
            value="",
        )

        baseline_evaluation_id = st.text_input(
            "Baseline evaluation ID",
            value="",
        )

        model_id = st.text_input(
            "Model ID",
            value="dinov3_gore",
        )

        dataset_id = st.text_input(
            "Dataset ID",
            value="",
        )

        subpath = st.text_input(
            "Subpath",
            value="",
        )

        device = st.selectbox(
            "Device",
            [
                "auto",
                "cpu",
                "cuda",
            ],
        )

        alpha = st.number_input(
            "Significance alpha",
            min_value=0.0001,
            max_value=0.9999,
            value=0.05,
            step=0.01,
        )

        force = st.checkbox(
            "Force rerun",
            value=False,
        )

        stop_on_failure = st.checkbox(
            "Stop on first experiment failure",
            value=False,
        )

        st.markdown("### Experiment builder")

        experiment_name = st.text_input("Spec name", value="confidence_threshold_sweep")
        hypothesis_claim = st.text_input("Hypothesis", value="A lower confidence threshold will improve recall without a large accuracy loss.")
        intervention_kind = st.selectbox("Intervention", ["inference", "image_transform", "preprocessing"], index=0)
        threshold_value = st.slider("Threshold value", 0.05, 0.95, 0.7, 0.01)
        fallback_class = st.text_input("Fallback class", value="unknown")
        analysis_id = st.text_input("Analysis ID (optional)", value="")
        experiment_config_valid = bool(experiment_name.strip())

        specs = []
        matrices = []

        if experiment_config_valid:
            if intervention_kind == "inference":
                specs.append(
                    make_inference_spec(
                        name=experiment_name,
                        claim=hypothesis_claim,
                        intervention_kind="inference",
                        threshold=threshold_value,
                        fallback_class=fallback_class,
                        analysis_id=analysis_id.strip() or None,
                    )
                )
            else:
                st.info("This UI currently provides the common inference-parameter experiment flow. More advanced graph/matrix experiments can still be submitted as raw JSON when needed.")

        if st.button(
            "▶ RUN EXPERIMENT FAMILY",
            type="primary",
            use_container_width=True,
            disabled=(
                not family_id.strip()
                or not experiment_config_valid
            ),
        ):

            with st.spinner(
                "Running controlled experiments..."
            ):

                result = api.experiments(
                    family_id=family_id.strip(),
                    baseline_evaluation_id=(
                        baseline_evaluation_id.strip()
                        or None
                    ),
                    specs=specs,
                    matrices=matrices,
                    model_id=(
                        model_id.strip()
                        or None
                    ),
                    dataset_id=(
                        dataset_id.strip()
                        or None
                    ),
                    subpath=(
                        subpath.strip()
                        or None
                    ),
                    device=device,
                    force=force,
                    stop_on_failure=stop_on_failure,
                    alpha=float(alpha),
                )

            st.session_state.last_response = result

            if result.get("ok"):

                st.success(
                    "Experiment family submitted successfully."
                )

            else:

                st.error(
                    "Experiment family failed."
                )

            show_response(result)


# ============================================================
# RESULTS
# ============================================================

elif page == "Results":

    st.title("📊 Results")

    st.caption(
        "Inspect the latest backend response and experiment outputs."
    )

    if st.session_state.last_response is None:

        st.info(
            "No result has been received during this UI session yet."
        )

    else:

        response = st.session_state.last_response

        if response.get("ok"):

            st.success("Latest operation completed/submitted.")

        else:

            st.error("Latest operation returned an error.")

        show_response(response)

        data = extract_data(response)

        if isinstance(data, dict):

            st.markdown("### Result summary")

            metrics = {}

            for key in [
                "accuracy",
                "macro_f1",
                "f1",
                "precision",
                "recall",
                "n",
                "n_errors",
                "num_samples",
                "num_tested_slices",
            ]:

                if key in data:
                    metrics[key] = data[key]

            if metrics:

                columns = st.columns(
                    min(
                        4,
                        max(1, len(metrics)),
                    )
                )

                for index, (key, value) in enumerate(
                    metrics.items()
                ):

                    with columns[index % len(columns)]:

                        st.metric(
                            key.replace("_", " ").title(),
                            str(value),
                        )


# ============================================================
# FAILURE EXPLORER
# ============================================================

elif page == "Failure Explorer":

    st.title("🔎 Failure Explorer")

    st.caption(
        "Inspect failure analysis output and discovered slices."
    )

    analysis_id = st.text_input(
        "Analysis ID",
        value="",
        help=(
            "Enter the analysis ID returned by Phase 4."
        ),
    )

    if analysis_id.strip():

        st.info(
            "The current Phase 1–5 API does not expose a dedicated "
            "GET /analyses/{analysis_id} contract in the server "
            "interface supplied for this UI. Use the Phase 4 "
            "response below or the Results page."
        )

    else:

        st.info(
            "Enter an analysis ID to identify the analysis you want "
            "to inspect."
        )


# ============================================================
# EXPERIMENTS
# ============================================================

elif page == "Experiments":

    st.title("🧪 Experiments")

    st.caption(
        "Phase 5 controlled experiment families."
    )

    st.info(
        "Experiment creation is available through "
        "Laboratory → Phase 5 — Controlled Experiments."
    )

    if st.session_state.last_response:

        data = extract_data(
            st.session_state.last_response
        )

        if isinstance(data, dict):

            family_id = data.get(
                "family_id"
            )

            if family_id:

                st.markdown(
                    f"### Last experiment family: "
                    f"`{family_id}`"
                )

            completed = data.get(
                "completed"
            )

            failed = data.get(
                "failed"
            )

            col1, col2 = st.columns(2)

            with col1:

                if completed is not None:

                    st.metric(
                        "Completed",
                        str(completed),
                    )

            with col2:

                if failed is not None:

                    st.metric(
                        "Failed",
                        str(failed),
                    )

            if "outcomes" in data:

                st.markdown("### Outcomes")

                st.json(
                    data["outcomes"]
                )

            if "report" in data:

                st.markdown("### Report")

                st.json(
                    data["report"]
                )


# ============================================================
# JOBS
# ============================================================

elif page == "Jobs":

    st.title("⚙️ Jobs")

    st.caption(
        "Monitor backend jobs."
    )

    if st.button(
        "Refresh jobs",
        use_container_width=True,
    ):

        st.rerun()

    jobs_response = api.jobs()

    if not jobs_response.get("ok"):

        st.error(
            "Could not retrieve jobs."
        )

        show_response(
            jobs_response
        )

        st.stop()

    jobs = extract_items(
        jobs_response,
        ("jobs", "items"),
    )

    if not jobs:

        st.info(
            "No jobs returned by the backend."
        )

    else:

        for job in jobs:

            if not isinstance(
                job,
                dict,
            ):
                continue

            job_id = (
                job.get("job_id")
                or job.get("id")
                or "unknown"
            )

            status = (
                job.get("status")
                or "unknown"
            )

            with st.expander(
                f"⚙️ {job_id} — {status}",
                expanded=False,
            ):

                st.json(job)

                if job_id != "unknown":

                    if st.button(
                        "Refresh this job",
                        key=f"refresh_job_{job_id}",
                    ):

                        job_response = api.job(
                            str(job_id)
                        )

                        st.json(
                            job_response
                        )

                    if st.button(
                        "Show log",
                        key=f"log_job_{job_id}",
                    ):

                        log_response = api.job_log(
                            str(job_id)
                        )

                        st.json(
                            log_response
                        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ModelLab • Deterministic experimental investigation engine"
)