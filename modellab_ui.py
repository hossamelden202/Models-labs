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
from pathlib import Path
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
        timeout: int = 3600,
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
            timeout=3600,
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
            timeout=3600,
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
# ============================================================
# API — CONTROLLED EXPERIMENTS
# Environment: Kaggle
# ============================================================

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
                "alpha": float(alpha),
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

    def get(self, path: str, params: dict | None = None) -> dict:
        return self._request(
            "GET",
            path,
            params=params,
            timeout=3600,
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
# FULL FAILURE ANALYSIS REPORT
# ============================================================


def _find_analysis_id(value):
    """
    Recursively find analysis_id in the different response
    shapes that ModelLab may return.
    """

    if isinstance(value, dict):

        direct = value.get("analysis_id")

        if direct:
            return str(direct)

        for key in (
            "result",
            "data",
            "response",
            "job",
        ):
            if key in value:
                found = _find_analysis_id(value[key])

                if found:
                    return found

    elif isinstance(value, list):

        for item in value:
            found = _find_analysis_id(item)

            if found:
                return found

    return None


def _render_report_value(value):

    if value is None:
        st.info("No data available.")
        return

    if isinstance(value, list):

        if not value:
            st.info("No entries.")
            return

        if all(isinstance(item, dict) for item in value):

            try:
                st.dataframe(
                    value,
                    use_container_width=True,
                    hide_index=True,
                )
                return
            except Exception:
                pass

        st.json(value)
        return

    if isinstance(value, dict):

        # Tables are much easier to read when possible.
        if value and all(
            isinstance(v, dict)
            for v in value.values()
        ):
            try:
                rows = []

                for key, item in value.items():
                    row = {"name": key}
                    row.update(item)
                    rows.append(row)

                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                )
                return
            except Exception:
                pass

        st.json(value)
        return

    st.write(value)


def _render_failure_analysis_report(report):

    if not isinstance(report, dict):
        st.error(
            "The failure-analysis endpoint returned an "
            "unexpected report format."
        )
        st.json(report)
        return

    st.markdown("---")
    st.markdown("## 🔬 Full Failure Analysis Report")

    st.caption(
        "Complete persisted failure-analysis report retrieved "
        "from the ModelLab backend."
    )

    # ========================================================
    # DOWNLOAD
    # ========================================================

    import json as _json

    report_json = _json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    download_name = (
        f"failure_analysis_"
        f"{report.get('inputs', {}).get('analysis_id', 'report')}.json"
    )

    st.download_button(
        label="⬇ Download Full Report (JSON)",
        data=report_json,
        file_name=download_name,
        mime="application/json",
        use_container_width=False,
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    overall = report.get("overall", {})
    inputs = report.get("inputs", {})

    st.markdown("### Overview")

    overview = [
        ("Samples", inputs.get("n_samples")),
        ("Image Accuracy", overall.get("image_accuracy")),
        ("Precision", overall.get("precision")),
        ("Recall", overall.get("recall")),
        ("F1", overall.get("f1")),
        ("True Positives", overall.get("true_positives")),
        ("False Positives", overall.get("false_positives")),
        ("False Negatives", overall.get("false_negatives")),
        ("Mean IoU", overall.get("mean_iou")),
        ("Correct Images", overall.get("correct_images")),
        ("Error Images", overall.get("error_images")),
    ]

    overview = [
        (label, value)
        for label, value in overview
        if value is not None
    ]

    if overview:

        columns = st.columns(
            min(4, len(overview))
        )

        for index, (label, value) in enumerate(overview):

            with columns[index % len(columns)]:

                if isinstance(value, float):
                    display = f"{value:.4f}"
                else:
                    display = str(value)

                st.metric(
                    label,
                    display,
                )

    # ========================================================
    # INPUTS
    # ========================================================

    with st.expander("Inputs", expanded=False):
        _render_report_value(inputs)

    # ========================================================
    # FAILURE TYPES
    # ========================================================

    if "failure_types" in report:

        with st.expander(
            "Failure Types",
            expanded=True,
        ):
            _render_report_value(
                report["failure_types"]
            )

    # ========================================================
    # PER CLASS
    # ========================================================

    if "per_class" in report:

        with st.expander(
            "Per Class",
            expanded=True,
        ):
            _render_report_value(
                report["per_class"]
            )

    # ========================================================
    # CONFIDENCE CALIBRATION
    # ========================================================

    if "confidence_calibration" in report:

        with st.expander(
            "Confidence Calibration",
            expanded=False,
        ):

            note = report.get(
                "confidence_calibration_note"
            )

            if note:
                st.info(str(note))

            _render_report_value(
                report["confidence_calibration"]
            )

    # ========================================================
    # THRESHOLD SENSITIVITY
    # ========================================================

    if "threshold_sensitivity" in report:

        with st.expander(
            "Threshold Sensitivity",
            expanded=False,
        ):
            _render_report_value(
                report["threshold_sensitivity"]
            )

    # ========================================================
    # LOCALIZATION
    # ========================================================

    if "localization" in report:

        with st.expander(
            "Localization",
            expanded=False,
        ):
            _render_report_value(
                report["localization"]
            )

    # ========================================================
    # OBJECT SIZE
    # ========================================================

    if "object_size" in report:

        with st.expander(
            "Object Size",
            expanded=False,
        ):
            _render_report_value(
                report["object_size"]
            )

    # ========================================================
    # EXTENDED SLICES
    # ========================================================

    if "extended_slices" in report:

        with st.expander(
            "Extended Slices",
            expanded=False,
        ):
            _render_report_value(
                report["extended_slices"]
            )

    # ========================================================
    # DIAGNOSTIC FINDINGS
    # ========================================================

    if "diagnostic_findings" in report:

        with st.expander(
            "Diagnostic Findings",
            expanded=True,
        ):
            _render_report_value(
                report["diagnostic_findings"]
            )

    # ========================================================
    # ACTIONABLE FINDINGS
    # ========================================================

    if "actionable_findings" in report:

        with st.expander(
            "Actionable Findings",
            expanded=True,
        ):
            _render_report_value(
                report["actionable_findings"]
            )

    # ========================================================
    # WORST SAMPLES
    # ========================================================

    if "worst_samples" in report:

        with st.expander(
            "Worst Samples",
            expanded=False,
        ):
            _render_report_value(
                report["worst_samples"]
            )

    # ========================================================
    # AUDIT CORRELATION
    # ========================================================

    if "audit_correlation" in report:

        with st.expander(
            "Audit Correlation",
            expanded=False,
        ):
            _render_report_value(
                report["audit_correlation"]
            )

    # ========================================================
    # LIMITATIONS
    # ========================================================

    if "limitations" in report:

        with st.expander(
            "Limitations",
            expanded=False,
        ):
            _render_report_value(
                report["limitations"]
            )

    # ========================================================
    # WARNINGS
    # ========================================================

    if "warnings" in report:

        with st.expander(
            "Warnings",
            expanded=False,
        ):
            _render_report_value(
                report["warnings"]
            )

    # ========================================================
    # RAW COMPLETE REPORT
    # ========================================================

    with st.expander(
        "📄 Raw Complete Report",
        expanded=False,
    ):
        st.json(report)


def _failure_analysis_render_job_result(api, job_result):
    """Render a persisted failure-analysis report embedded in a completed job response."""

    if not isinstance(job_result, dict):
        return False

    report = None

    for key in ("analysis_report", "report"):
        candidate = job_result.get(key)

        if isinstance(candidate, dict):
            report = candidate
            break

    if report is None:
        data = job_result.get("data")

        if isinstance(data, dict):
            for key in ("analysis_report", "report"):
                candidate = data.get(key)

                if isinstance(candidate, dict):
                    report = candidate
                    break

    if report is None:
        return False

    _render_failure_analysis_report(report)

    with st.expander(
        "📄 Raw job payload",
        expanded=False,
    ):
        st.json(job_result)

    return True


def _load_full_failure_analysis_report(api, analysis_id):

    if not analysis_id:
        raise RuntimeError(
            "No analysis ID was returned by the backend."
        )

    # The backend already exposes:
    #
    # GET /analyses/{analysis_id}
    #
    return api.get(
        f"/analyses/{analysis_id}"
    )


def _find_local_failure_analysis_reports() -> list[dict[str, Any]]:
    """List saved failure-analysis reports from the local artifacts folder."""

    candidates = [
        Path("/kaggle/working/modellab/workspace/artifacts/failure_analyses"),
        Path("/kaggle/working/modellab/artifacts/failure_analyses"),
        Path("/kaggle/working/artifacts/failure_analyses"),
        Path("artifacts/failure_analyses"),
        Path.cwd() / "artifacts" / "failure_analyses",
    ]

    reports: list[dict[str, Any]] = []
    seen: set[str] = set()

    for root in candidates:
        if not root.exists():
            continue

        for report_path in sorted(root.glob("*/report.json")):
            try:
                with report_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            inputs = payload.get("inputs") or {}
            evaluation_id = str(
                inputs.get("evaluation_id")
                or report_path.parent.name
            )

            report_key = f"{evaluation_id}:{report_path.parent.name}"
            if report_key in seen:
                continue

            seen.add(report_key)
            reports.append(
                {
                    "evaluation_id": evaluation_id,
                    "analysis_id": str(
                        payload.get("inputs", {}).get("analysis_id")
                        or report_path.parent.name
                    ),
                    "report": payload,
                    "path": str(report_path),
                }
            )

    return sorted(
        reports,
        key=lambda item: (
            str(item["evaluation_id"]),
            str(item["analysis_id"]),
        ),
    )

# ============================================================
# EXPERIMENT CAMPAIGN HELPERS
# ============================================================

def _find_campaign(value):
    """Find the dict that holds 'outcomes' (the campaign) in any response shape."""
    if isinstance(value, dict):
        if isinstance(value.get("outcomes"), list):
            return value
        for key in ("result", "data", "response", "job", "payload"):
            if key in value:
                found = _find_campaign(value[key])
                if found:
                    return found
    return None


def _strip_nulls(value):
    if isinstance(value, dict):
        return {k: _strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_nulls(v) for v in value if v is not None]
    return value


def _render_flexible(value):
    """Render str / list of str / list of dicts / dict."""
    if value is None or value == "" or value == [] or value == {}:
        st.info("Nothing returned.")
    elif isinstance(value, str):
        st.write(value)
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        for item in value:
            st.markdown(f"- {item}")
    else:
        _render_report_value(value)


def _render_scalar_metrics(section):
    if not isinstance(section, dict):
        return
    scalars = {
        k: v for k, v in section.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if not scalars:
        return
    cols = st.columns(min(4, len(scalars)))
    for i, (k, v) in enumerate(scalars.items()):
        with cols[i % len(cols)]:
            st.metric(
                k.replace("_", " ").title(),
                f"{v:.4f}" if isinstance(v, float) else str(v),
            )


def _render_training(training):
    if not isinstance(training, dict):
        _render_flexible(training)
        return

    _render_scalar_metrics(training)

    if training.get("stopped_early") is not None:
        st.caption(f"Stopped early: **{training['stopped_early']}**")

    for key in ("best_checkpoint", "last_checkpoint", "run_dir"):
        if training.get(key):
            st.caption(f"{key.replace('_', ' ').title()}: `{training[key]}`")

    history = training.get("history")
    if (
        isinstance(history, list)
        and history
        and all(isinstance(h, dict) for h in history)
    ):
        import pandas as pd

        df = pd.DataFrame(history)
        if "epoch" in df.columns:
            df = df.set_index("epoch")

        metric_cols = [c for c in df.columns if str(c).startswith("metrics/")]
        loss_cols = [c for c in df.columns if "loss" in str(c)]

        if metric_cols:
            st.markdown("**Validation metrics per epoch**")
            st.line_chart(df[metric_cols])

        if loss_cols:
            st.markdown("**Losses per epoch**")
            st.line_chart(df[loss_cols])

        with st.expander("History table", expanded=False):
            st.dataframe(df, use_container_width=True)

    config = training.get("config")
    if isinstance(config, dict):
        with st.expander("Training config used (nulls hidden)", expanded=False):
            st.json(_strip_nulls(config))


def _render_outcome(outcome):
    if not isinstance(outcome, dict):
        st.json(outcome)
        return

    result = outcome.get("result") or {}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Experiment ID", str(outcome.get("experiment_id", "—")))
    with c2:
        st.metric("Status", str(outcome.get("status", "—")))
    with c3:
        st.metric("Task", str(result.get("task", "—")))

    if result.get("checkpoint"):
        st.caption(f"Checkpoint: `{result['checkpoint']}`")

    if outcome.get("error") or result.get("error"):
        st.error(str(outcome.get("error") or result.get("error")))

    tab_train, tab_eval, tab_cmp, tab_raw = st.tabs(
        ["Training", "Evaluation", "Comparison", "Raw"]
    )

    with tab_train:
        _render_training(result.get("training"))

    with tab_eval:
        _render_scalar_metrics(result.get("evaluation"))
        _render_flexible(result.get("evaluation"))

    with tab_cmp:
        _render_scalar_metrics(result.get("comparison"))
        _render_flexible(result.get("comparison"))

    with tab_raw:
        st.json(outcome)


def _render_campaign(campaign):
    outcomes = campaign.get("outcomes") or []

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Campaign ID", str(campaign.get("campaign_id", "—")))
    with c2:
        st.metric("Family ID", str(campaign.get("family_id", "—")))
    with c3:
        st.metric("Experiments", len(outcomes))

    if campaign.get("directory"):
        st.caption(f"Directory: `{campaign['directory']}`")

    tab_sum, tab_exp, tab_stages, tab_find, tab_report, tab_raw = st.tabs(
        ["Summary", "Experiments", "Stages", "Findings", "Report", "Raw"]
    )

    with tab_sum:
        st.markdown("### Best experiment")
        _render_flexible(campaign.get("best_experiment"))
        st.markdown("### Final recommendation")
        _render_flexible(campaign.get("final_recommendation"))

    with tab_exp:
        if not outcomes:
            st.info("No experiment outcomes in this campaign.")
        else:
            def _label(i):
                o = outcomes[i] if isinstance(outcomes[i], dict) else {}
                return (
                    f"{o.get('name', 'experiment')} — "
                    f"{o.get('experiment_id', 'unknown')} — "
                    f"{o.get('status', 'unknown')}"
                )

            chosen = st.selectbox(
                "Select experiment",
                list(range(len(outcomes))),
                format_func=_label,
                key="results_experiment_select",
            )
            _render_outcome(outcomes[chosen])

    with tab_stages:
        _render_flexible(campaign.get("stages"))

    with tab_find:
        _render_flexible(campaign.get("findings"))

    with tab_report:
        _render_flexible(campaign.get("report"))

    with tab_raw:
        st.json(campaign)

# ============================================================
# EXPERIMENT FAMILY: AUTO FETCH + TABS
# ============================================================

def _fetch_family_bundle(family_id):
    """GET family + every experiment detail + paired. Read-only."""
    family = api.get(f"/experiments/{family_id}")
    bundle = {
        "family_id": family_id,
        "family": family,
        "details": {},
        "paired": {},
        "error": None,
    }

    if not family.get("ok"):
        bundle["error"] = family
    else:
        for exp in family.get("experiments") or []:
            if not isinstance(exp, dict):
                continue
            eid = exp.get("experiment_id")
            if not eid:
                continue
            bundle["details"][eid] = api.get(f"/experiments/{family_id}/{eid}")
            bundle["paired"][eid] = api.get(f"/experiments/{family_id}/{eid}/paired")

    st.session_state.experiment_bundle = bundle
    st.session_state["last_family_id"] = family_id
    return bundle


def _comparison_table(comparison):
    """Baseline / Trained / Delta table. Works for any metric names."""
    if not isinstance(comparison, dict):
        return False

    baseline = comparison.get("baseline") or {}
    trained = comparison.get("trained") or {}
    delta = comparison.get("delta") or {}

    keys = []
    for src in (baseline, trained, delta):
        if isinstance(src, dict):
            for k, v in src.items():
                if (
                    k not in keys
                    and isinstance(v, (int, float))
                    and not isinstance(v, bool)
                ):
                    keys.append(k)

    if not keys:
        return False

    rows = [
        {
            "Metric": k.replace("_", " ").title(),
            "Baseline": baseline.get(k),
            "Trained": trained.get(k),
            "Delta": delta.get(k),
        }
        for k in keys
    ]

    import pandas as pd

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )
    return True


def _render_family_experiment(exp, detail, paired):
    status = str(exp.get("status", "unknown"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Experiment ID", str(exp.get("experiment_id", "—")))
    with c2:
        st.metric("Name", str(exp.get("name", "—")))
    with c3:
        st.metric("Status", status)

    if exp.get("error"):
        st.error(str(exp["error"]))

    if status.lower() not in {"completed", "success"}:
        st.info(f"This experiment is `{status}`. Results appear when it completes.")

    verdict = exp.get("verdict")
    if verdict:
        st.markdown("**Verdict**")
        _render_flexible(verdict)

    t_cmp, t_eval, t_class, t_int, t_det, t_pair, t_raw = st.tabs(
        [
            "Comparison",
            "Evaluation",
            "Per-class",
            "Intervention",
            "Details / Training",
            "Paired",
            "Raw",
        ]
    )

    with t_cmp:
        comparison = exp.get("comparison")
        if exp.get("accuracy_difference") is not None:
            st.metric("Accuracy difference", f"{exp['accuracy_difference']:+.4f}")
        if not _comparison_table(comparison):
            _render_flexible(comparison)

    with t_eval:
        evaluation = exp.get("evaluation")
        if isinstance(evaluation, dict):
            if evaluation.get("evaluation_id"):
                st.caption(f"Evaluation ID: `{evaluation['evaluation_id']}`")
            _render_scalar_metrics(evaluation)
            summary = {
                k: v for k, v in evaluation.items()
                if k not in {"per_class"}
                and not isinstance(v, (int, float, str, bool, type(None)))
            }
            if summary:
                _render_report_value(summary)
        else:
            _render_flexible(evaluation)

    with t_class:
        evaluation = exp.get("evaluation")
        per_class = evaluation.get("per_class") if isinstance(evaluation, dict) else None
        if per_class:
            _render_report_value(per_class)
        else:
            st.info("No per-class metrics returned.")

    with t_int:
        _render_flexible(_strip_nulls(exp.get("intervention")))

    with t_det:
        if isinstance(detail, dict) and detail.get("ok"):
            data = extract_data(detail)
            training = None
            if isinstance(data, dict):
                res = data.get("result") if isinstance(data.get("result"), dict) else data
                training = res.get("training") if isinstance(res, dict) else None
            if training:
                _render_training(training)
            else:
                _render_flexible(_strip_nulls(data))
        else:
            st.warning("Experiment detail not available.")
            st.json(detail)

    with t_pair:
        if isinstance(paired, dict) and paired.get("ok"):
            _render_flexible(_strip_nulls(extract_data(paired)))
        else:
            st.info("No paired results for this experiment.")
            if paired:
                with st.expander("Response", expanded=False):
                    st.json(paired)

    with t_raw:
        st.json(exp)


def _render_family_bundle(bundle):
    if bundle.get("error"):
        st.error(f"Could not load family `{bundle['family_id']}` (GET /experiments/{{family_id}}).")
        st.json(bundle["error"])
        return

    family = bundle["family"]
    report = family.get("family_report") or {}
    experiments = [e for e in (family.get("experiments") or []) if isinstance(e, dict)]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Family", str(report.get("family_id", bundle["family_id"])))
    with c2:
        st.metric("Task", str(report.get("task", "—")).title())
    with c3:
        st.metric("Experiments", str(report.get("num_specs", len(experiments))))
    with c4:
        st.metric("Completed", str(report.get("num_completed", "—")))
    with c5:
        st.metric("Failed", str(report.get("num_failed", "—")))

    tab_exp, tab_over, tab_base, tab_camp, tab_raw = st.tabs(
        ["Experiments", "Overview", "Baseline", "Campaigns", "Raw"]
    )

    with tab_exp:
        if not experiments:
            st.info("No experiments in this family.")
        else:
            def _label(i):
                e = experiments[i]
                return f"{e.get('name', 'experiment')} — {e.get('experiment_id', '?')} — {e.get('status', '?')}"

            chosen = st.selectbox(
                "Select experiment",
                list(range(len(experiments))),
                format_func=_label,
                key="family_exp_select",
            )
            exp = experiments[chosen]
            eid = exp.get("experiment_id")
            _render_family_experiment(
                exp,
                bundle["details"].get(eid),
                bundle["paired"].get(eid),
            )

    with tab_over:
        if experiments:
            import pandas as pd

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "experiment_id": e.get("experiment_id"),
                            "name": e.get("name"),
                            "status": e.get("status"),
                            "accuracy_difference": e.get("accuracy_difference"),
                            "verdict": str(e.get("verdict")) if e.get("verdict") is not None else None,
                        }
                        for e in experiments
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        _render_report_value(report)

        outcomes = family.get("outcomes")
        if outcomes:
            with st.expander("Outcomes", expanded=False):
                st.json(outcomes)

    with tab_base:
        _render_flexible(_strip_nulls(family.get("baseline")))

    with tab_camp:
        campaigns = family.get("campaigns") or []
        if not campaigns:
            st.info("No campaigns exist for this family.")
        else:
            for camp in campaigns:
                if isinstance(camp, dict) and isinstance(camp.get("outcomes"), list):
                    _render_campaign(camp)
                else:
                    st.json(camp)

    with tab_raw:
        st.json(family)


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
# MODEL CONFIG IMPORT
# ============================================================

def import_config_upload(uploaded_file):
    """Parse an uploaded .json/.yaml/.yml. Returns (spec_fields, error)."""
    import os
    import sys

    try:
        try:
            from modellab.config_import import (
                ConfigImportError,
                import_model_config,
            )
        except ImportError:
            src = Path(__file__).resolve().parent / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            from modellab.config_import import (
                ConfigImportError,
                import_model_config,
            )
    except Exception as exc:
        return None, f"Config importer is not importable: {exc}"

    suffix = Path(uploaded_file.name).suffix.lower() or ".json"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded_file.getvalue())
            tmp_path = handle.name

        result = import_model_config(tmp_path)  # <-- change here if it takes text/bytes
        return result.to_spec_fields(), None

    except ConfigImportError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Unexpected import error: {exc}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class _PathUpload:
    """Makes a file on disk look like a Streamlit upload."""

    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data

def import_config_path(path_text):
    """Ask the BACKEND to read + parse a config at a path on its machine."""
    response = api._request(
        "POST",
        "/config/import",
        json_body={"path": path_text.strip()},
        timeout=60,
    )

    if response.get("ok"):
        fields = response.get("spec_fields")
        if isinstance(fields, dict):
            return fields, None
        return None, "Backend returned no spec_fields."

    detail = response.get("detail") or response.get("error") or response
    return None, str(detail)


# ============================================================
# MODELS
# ============================================================

if page == "Models":
    st.title("🧠 Models")

    # ============================================================
    # 1. REGISTERED MODELS
    # ============================================================

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

    # ============================================================
    # 2. MODEL REGISTRATION
    # ============================================================

    st.markdown("### Register a model")

    st.caption(
        "Stage the artifact first, then let the backend inspect and "
        "resolve the model. Metadata fields are optional overrides; "
        "when left empty, the backend infers them from the artifact."
    )

    # ---------------- Optional config import ----------------
    st.markdown("#### Import model config (optional)")

    config_source = st.radio(
        "Config source",
        ["None", "Upload file", "Backend path (Kaggle)"],
        horizontal=True,
        key="model_config_source",
    )

    imported_cfg = None
    import_error = None

    if config_source == "Upload file":
        config_file = st.file_uploader(
            "Config file (.json / .yaml / .yml)",
            type=["json", "yaml", "yml"],
            key="model_config_import",
            help=(
                "Generic JSON/YAML, Ultralytics YAML or Hugging Face config.json. "
                "Anything you type in the override fields below wins."
            ),
        )
        if config_file is not None:
            imported_cfg, import_error = import_config_upload(config_file)

    elif config_source == "Backend path (Kaggle)":
        config_path_text = st.text_input(
            "Config file path",
            value="",
            key="model_config_path",
            placeholder="/kaggle/input/my-dataset/data.yaml",
            help="Read by the backend on Kaggle, not on this PC.",
        )
        if config_path_text.strip():
            imported_cfg, import_error = import_config_path(config_path_text)

    if import_error:
        st.error(f"Config import failed: {import_error}")
    elif imported_cfg:
        st.success("Config imported.")
        with st.expander("Imported values", expanded=True):
            st.json(imported_cfg)
    source_kind = st.radio(
        "Source",
        [
            "Upload",
            "Server path",
            "Existing artifact",
        ],
        horizontal=True,
        key="model_source_kind",
    )

    with st.form("register_model_form"):

        model_id = st.text_input(
            "Model ID",
            value="demo_model",
            help="Identifier used when the resolved model is registered.",
        )



        # ========================================================
        # SOURCE
        # ========================================================

        if source_kind == "Existing artifact":

            artifact_id = st.text_input(
                "Artifact ID",
                value="",
                help=(
                    "Use an artifact that is already staged by the "
                    "backend."
                ),
            )

            local_path = ""
            artifact_file = None

        elif source_kind == "Server path":

            artifact_id = ""

            local_path = st.text_input(
                "Server path",
                value="",
                help=(
                    "Path visible to the backend, for example a "
                    "Kaggle input or workspace path."
                ),
            )

            artifact_file = None

        else:

            artifact_id = ""
            local_path = ""

            artifact_file = st.file_uploader(
                "Artifact file",
                type=[
                    "pt",
                    "pth",
                    "ckpt",
                    "bin",
                    "safetensors",
                    "onnx",
                    "pb",
                    "keras",
                    "py",
                ],
                help=(
                    "Upload the model artifact. The backend will inspect "
                    "it before attempting model resolution."
                ),
            )

        factory_file = st.file_uploader(
            "Factory file (optional)",
            type=["py"],
            help=(
                "Only needed when the model must be constructed through "
                "a custom Python factory."
            ),
        )

        # ========================================================
        # OPTIONAL BACKEND OVERRIDES
        # ========================================================

        st.markdown("#### Optional metadata overrides")

        st.caption(
            "Leave fields empty to let the backend infer them. "
            "Do not enter placeholder values unless you intentionally "
            "want to override the artifact metadata."
        )

        architecture_hint = st.text_input(
            "Architecture hint",
            value="",
            help=(
                "Optional architecture hint. Leave empty unless you "
                "specifically want to guide resolution."
            ),
        )

        class_names_text = st.text_input(
            "Class names",
            value="",
            help=(
                "Optional comma-separated class names. Leave empty "
                "when the checkpoint contains its own class names."
            ),
        )

        num_classes = st.number_input(
            "Number of classes override",
            min_value=1,
            max_value=1000,
            value=None,
            step=1,
            placeholder="Auto-detect",
            help=(
                "Optional. Leave empty to use the number of classes "
                "reported by the backend/model."
            ),
        )

        input_width = st.number_input(
            "Input width override",
            min_value=1,
            max_value=4096,
            value=None,
            step=1,
            placeholder="Auto-detect",
            help=(
                "Optional. Leave empty to use the model/backend "
                "metadata when available."
            ),
        )

        input_height = st.number_input(
            "Input height override",
            min_value=1,
            max_value=4096,
            value=None,
            step=1,
            placeholder="Auto-detect",
            help=(
                "Optional. Leave empty to use the model/backend "
                "metadata when available."
            ),
        )

        state_dict_key = st.text_input(
            "State dict key",
            value="",
            help=(
                "Optional. Only use this for checkpoints containing "
                "multiple nested state dictionaries."
            ),
        )

        # ========================================================
        # SAFETY / VALIDATION
        # ========================================================

        trust_executable = st.checkbox(
            "I trust executable model code",
            value=False,
            help=(
                "Required for trusted full-model/pickle checkpoints "
                "such as many Ultralytics .pt files."
            ),
        )

        validate_load = st.checkbox(
            "Validate model load",
            value=True,
            help=(
                "Run the backend model-load and forward-pass validation "
                "when enough information is available."
            ),
        )

        submit_model = st.form_submit_button(
            "Inspect + Register Model",
            type="primary",
            use_container_width=True,
        )

    # ============================================================
    # 3. SUBMIT
    # ============================================================

    if submit_model:

        try:

            files: dict[str, Any] = {}
            handles: list[Any] = []

            try:

                # ====================================================
                # 3A. PREPARE UPLOADS
                # ====================================================

                if artifact_file is not None:

                    weight_path = temp_file_from_upload(
                        artifact_file,
                        ".bin",
                    )

                    if weight_path:

                        handle = open(
                            weight_path,
                            "rb",
                        )

                        handles.append(handle)

                        files["artifact"] = (
                            artifact_file.name or "artifact.bin",
                            handle,
                        )

                if factory_file is not None:

                    factory_path = temp_file_from_upload(
                        factory_file,
                        ".py",
                    )

                    if factory_path:

                        handle = open(
                            factory_path,
                            "rb",
                        )

                        handles.append(handle)

                        files["factory_file"] = (
                            factory_file.name or "factory.py",
                            handle,
                        )

                # ====================================================
                # 3B. BUILD STAGING REQUEST
                # ====================================================

                stage_data: dict[str, str] = {}

                if source_kind == "Existing artifact":

                    if artifact_id.strip():

                        stage_data["artifact_id"] = (
                            artifact_id.strip()
                        )

                elif source_kind == "Server path":

                    if local_path.strip():

                        stage_data["local_path"] = (
                            local_path.strip()
                        )

                # ====================================================
                # 3C. STAGE / INSPECT ARTIFACT
                # ====================================================

                inspection_response = api._request(
                    "POST",
                    "/model-artifacts",
                    data=stage_data,
                    files=files,
                    timeout=300,
                )

                if not inspection_response.get("ok"):

                    st.error(
                        "Artifact staging/inspection failed."
                    )

                    show_response(
                        inspection_response
                    )

                    st.stop()

                # ====================================================
                # 3D. EXTRACT BACKEND STAGING PAYLOAD
                # ====================================================

                staged_payload = (
                    inspection_response.get("payload")
                )

                if not isinstance(
                    staged_payload,
                    dict,
                ):

                    staged_payload = inspection_response

                staged_artifact = (
                    staged_payload.get("artifact")
                )

                if not isinstance(
                    staged_artifact,
                    dict,
                ):

                    staged_artifact = {}

                staged_artifact_id = (
                    staged_payload.get("artifact_id")
                    or staged_artifact.get("artifact_id")
                )

                if not staged_artifact_id:

                    st.error(
                        "The backend successfully staged the artifact, "
                        "but did not return an artifact_id."
                    )

                    st.json(
                        inspection_response
                    )

                    st.stop()

                # IMPORTANT:
                # After staging, always use the backend-issued ID.
                staged_artifact_id = str(
                    staged_artifact_id
                )

                staged_inspection = (
                    staged_payload.get("inspection")
                    or inspection_response.get("inspection")
                    or {}
                )

                # ====================================================
                # 3E. SHOW INSPECTION
                # ====================================================

                with st.expander(
                    "🔍 Backend artifact inspection",
                    expanded=False,
                ):

                    st.json(
                        {
                            "artifact_id": staged_artifact_id,
                            "inspection": staged_inspection,
                        }
                    )

                # ====================================================
                # 3F. BUILD RESOLUTION REQUEST
                # ====================================================

                payload = artifact_resolution_defaults(
                    artifact_id=staged_artifact_id,
                    inspection=staged_inspection,
                    preferred_architecture=(
                        architecture_hint.strip()
                        or None
                    ),
                )

                if not isinstance(
                    payload,
                    dict,
                ):

                    payload = {
                        "artifact_id": staged_artifact_id,
                    }

                # IMPORTANT:
                # Do NOT inject fake/default metadata.
                # Only add values explicitly supplied by the user.
                # ----------------------------------------------------

                # Imported config: overrides backend hints, but is
                # itself overridden by the manual fields below.
                if imported_cfg:
                    for _key in ("num_classes", "class_names", "input_size"):
                        if imported_cfg.get(_key) is not None:
                            payload[_key] = imported_cfg[_key]

                if class_names_text.strip():

                    payload["class_names"] = (
                        parse_csv_values(
                            class_names_text
                        )
                    )

                if num_classes is not None:

                    payload["num_classes"] = int(
                        num_classes
                    )

                if (
                    input_width is not None
                    and input_height is not None
                ):

                    payload["input_size"] = [
                        int(input_width),
                        int(input_height),
                    ]

                elif (
                    input_width is not None
                    or input_height is not None
                ):

                    st.error(
                        "Input width and input height must either "
                        "both be provided or both be left empty."
                    )

                    st.stop()

                if state_dict_key.strip():

                    payload["state_dict_key"] = (
                        state_dict_key.strip()
                    )

                if trust_executable:

                    payload["trust_executable"] = True

                payload["validate_load"] = bool(
                    validate_load
                )

                # ====================================================
                # 3G. SHOW ACTUAL REQUEST
                # ====================================================

                with st.expander(
                    "📤 Model resolution request",
                    expanded=False,
                ):

                    st.json(payload)

                # ====================================================
                # 3H. RESOLVE
                # ====================================================

                resolution = api._request(
                    "POST",
                    "/models/resolve",
                    json_body=payload,
                    timeout=300,
                )

                st.session_state.last_response = (
                    resolution
                )

                # ====================================================
                # 3I. RESOLUTION RESULT
                # ====================================================

                if resolution.get("status") == "ready":

                    st.success(
                        "Model resolution succeeded."
                    )

                    # Show exactly what the backend resolved.
                    st.markdown(
                        "#### Backend-resolved model"
                    )

                    resolution_payload = (
                        resolution.get("payload")
                        if isinstance(
                            resolution.get("payload"),
                            dict,
                        )
                        else resolution
                    )

                    resolved_spec = (
                        resolution_payload.get("spec")
                        if isinstance(
                            resolution_payload,
                            dict,
                        )
                        else None
                    )

                    if isinstance(
                        resolved_spec,
                        dict,
                    ):

                        st.json(
                            {
                                "model_id": (
                                    resolved_spec.get(
                                        "model_id"
                                    )
                                ),
                                "task": (
                                    resolved_spec.get(
                                        "task"
                                    )
                                ),
                                "num_classes": (
                                    resolved_spec.get(
                                        "num_classes"
                                    )
                                ),
                                "class_names": (
                                    resolved_spec.get(
                                        "class_names"
                                    )
                                ),
                                "input_size": (
                                    resolved_spec.get(
                                        "input_size"
                                    )
                                ),
                                "source": (
                                    resolved_spec.get(
                                        "source"
                                    )
                                ),
                            }
                        )

                    # =================================================
                    # 3J. REGISTER
                    # =================================================

                    # =================================================
                    # 3J. REGISTER RESOLVED MODEL
                    # =================================================
                    #
                    # /models/from-artifact is a resolution endpoint,
                    # not the model registration endpoint.
                    #
                    # /models/register expects:
                    #   {
                    #       "spec": <resolved model spec>,
                    #       "preprocess": <preprocess config>,
                    #       "validate_load": <bool>,
                    #       "model_id": <id>
                    #   }
                    #
                    # Use the backend's resolved spec so registration
                    # receives the authoritative model metadata.

                    if not isinstance(resolved_spec, dict):
                        st.error(
                            "The backend resolved the model, but did not "
                            "return a valid model specification."
                        )

                        show_response(resolution)
                        st.stop()

                    # =================================================
                    # 3J. REGISTER RESOLVED MODEL
                    # =================================================
                    #
                    # The resolver response contains resolver-only fields
                    # such as "task".  The /models/register endpoint uses
                    # the TorchModelSpec contract and currently rejects
                    # unknown fields.  Build the registration spec
                    # explicitly instead of forwarding the entire
                    # resolver response.

                    if not isinstance(resolved_spec, dict):
                        st.error(
                            "The backend resolved the model, but did not "
                            "return a valid model specification."
                        )

                        show_response(resolution)
                        st.stop()

                    registration_spec = {}

                    for key in (
                        "source",
                        "task",
                        "num_classes",
                        "path",
                        "factory",
                        "factory_kwargs",
                        "state_dict_key",
                        "strip_prefix",
                        "class_names",
                        "input_size",
                        "output_key",
                        "output_index",
                        "single_logit_binary",
                        "mixed_precision",
                        "architecture",
                        "architecture_config",
                    ):
                        if key in resolved_spec:
                            registration_spec[key] = resolved_spec[key]

                    registration_payload = {
                        "spec": registration_spec,
                        "preprocess": (
                            resolution_payload.get("preprocess")
                            if isinstance(resolution_payload.get("preprocess"), dict)
                            else {}
                        ),
                        "validate_load": bool(validate_load),
                        "model_id": model_id.strip() or staged_artifact_id,
                    }

                    with st.expander("📥 Model registration request", expanded=False):
                        st.json(registration_payload)

                    registration = api._request(
                        "POST",
                        "/models/register",
                        json_body=registration_payload,
                        timeout=300,
                    )

                    st.session_state.last_response = (
                        registration
                    )

                    if registration.get("ok"):

                        st.success(
                            f"Model `{model_id.strip() or staged_artifact_id}` "
                            "registered successfully."
                        )

                        show_response(
                            registration
                        )

                    else:

                        st.error(
                            "Resolution succeeded, "
                            "but model registration failed."
                        )

                        show_response(
                            registration
                        )

                else:

                    st.warning(
                        "The backend could not resolve the model yet."
                    )

                    st.json(
                        {
                            "staged_artifact_id": (
                                staged_artifact_id
                            ),
                            "request": payload,
                            "resolution": resolution,
                        }
                    )

                    show_response(
                        resolution
                    )

            finally:

                for handle in handles:

                    try:
                        handle.close()

                    except Exception:
                        pass

        except Exception as exc:

            st.error(
                f"Invalid model configuration: {exc}"
            )
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

                                # Failure-analysis jobs now return the complete persisted
                                # report directly from GET /jobs/{job_id}.
                                if isinstance(job_data, dict):
                                    analysis_report = job_data.get("analysis_report")

                                    if isinstance(analysis_report, dict):
                                        st.session_state["last_analysis_report"] = analysis_report

                                progress.progress(1.0)
                                status_box.success(
                                    "Dataset audit completed."
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
            "high_confidence_threshold": st.slider(
                "High confidence threshold",
                0.5,
                1.0,
                0.9,
                0.01,
            ),
            "low_confidence_threshold": st.slider(
                "Low confidence threshold",
                0.1,
                1.0,
                0.6,
                0.01,
            ),
            "min_support": st.number_input(
                "Minimum support",
                min_value=2,
                max_value=1000,
                value=20,
                step=1,
            ),
            "alpha": st.slider(
                "Significance alpha",
                0.01,
                0.2,
                0.05,
                0.01,
            ),
            "max_evidence": st.number_input(
                "Max evidence",
                min_value=1,
                max_value=200,
                value=20,
                step=1,
            ),
            "max_listed_slices": st.number_input(
                "Max listed slices",
                min_value=1,
                max_value=500,
                value=100,
                step=1,
            ),
            "top_confusions": st.number_input(
                "Top confusions",
                min_value=1,
                max_value=50,
                value=5,
                step=1,
            ),
            "min_risk_difference": st.slider(
                "Min risk difference",
                0.0,
                0.5,
                0.05,
                0.01,
            ),
        }

        analysis_config = make_analysis_config(
            **analysis_config
        )

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

            # ------------------------------------------------
            # GET ANALYSIS ID
            # ------------------------------------------------

            analysis_id_value = _find_analysis_id(
                result
            )

            if analysis_id_value:

                st.session_state.last_analysis_id = (
                    analysis_id_value
                )

                # ------------------------------------------------
                # RETRIEVE FULL PERSISTED REPORT
                # ------------------------------------------------

                try:

                    with st.spinner(
                        "Retrieving the complete failure-analysis report..."
                    ):

                        full_report = (
                            _load_full_failure_analysis_report(
                                api,
                                analysis_id_value,
                            )
                        )

                    st.session_state.last_analysis_report = (
                        full_report
                    )

                    st.success(
                        "Failure analysis completed successfully."
                    )

                    _render_failure_analysis_report(
                        full_report
                    )

                except Exception as exc:

                    st.error(
                        "The analysis completed, but the full "
                        "failure-analysis report could not be retrieved."
                    )

                    st.exception(exc)

            elif result.get("ok"):

                st.success(
                    "Failure analysis submitted successfully, "
                    "but no analysis ID was returned yet."
                )

                st.info(
                    "The compact backend response is shown below."
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

        st.subheader("Controlled Experiments")
        st.caption("Run controlled training experiments against a persisted baseline.")

        family_id = st.text_input(
            "Experiment family ID",
            value="",
            help="Example: weapons-final-model",
            key="p5_family_id",
        )

        baseline_evaluation_id = st.text_input(
            "Baseline Evaluation ID",
            value="",
            help="Existing evaluation ID used as the baseline for comparison.",
            key="p5_baseline_evaluation_id",
        )

        model_id = st.text_input(
            "Model ID",
            value="weapons-final-model",
            key="p5_model_id",
        )

        dataset_id = st.text_input(
            "Dataset ID",
            value="weapons-final",
            key="p5_dataset_id",
        )

        alpha = st.number_input(
            "Significance alpha",
            min_value=0.0001,
            max_value=0.9999,
            value=0.05,
            step=0.01,
            key="p5_alpha",
        )

        force = st.checkbox("Force rerun", value=False, key="p5_force")
        stop_on_failure = st.checkbox(
            "Stop on first experiment failure", value=False, key="p5_stop"
        )

        st.markdown("### Detection Training Experiment")

        experiment_name = st.text_input(
            "Experiment name",
            value="detection_training_experiment",
            key="p5_name",
        )

        hypothesis_claim = st.text_input(
            "Hypothesis",
            value=(
                "Training the detector with the selected configuration "
                "will improve detection F1 over the baseline."
            ),
            key="p5_claim",
        )

        # ---------- helper: optional number (checkbox = "send a value") ----------
        def _opt_num(label, key, **kwargs):
            c_on, c_val = st.columns([1, 2])
            with c_on:
                on = st.checkbox(f"Set {label}", value=False, key=f"{key}_on")
            with c_val:
                val = st.number_input(label, key=key, disabled=not on, **kwargs)
            return val if on else None

        cfg_core, cfg_aug, cfg_opt, cfg_loss, cfg_adv = st.tabs(
            ["Core", "Augmentation", "Optimizer & Scheduler", "Loss", "Advanced"]
        )

        # ---------------- CORE ----------------
        with cfg_core:
            k1, k2 = st.columns(2)
            with k1:
                epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=1, step=1, key="p5_epochs")
                p5_batch_size = st.number_input("Batch size", min_value=1, max_value=512, value=32, step=1, key="p5_batch")
                p5_input_width = st.number_input("Input width", min_value=32, max_value=2048, value=224, step=32, key="p5_w")
                p5_seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="p5_seed")
            with k2:
                p5_num_workers = st.number_input("Workers", min_value=0, max_value=32, value=2, step=1, key="p5_workers")
                p5_input_height = st.number_input("Input height", min_value=32, max_value=2048, value=224, step=32, key="p5_h")
                sampler = st.selectbox("Sampler", ["default", "balanced"], key="p5_sampler")
                monitor = st.text_input(
                    "Monitor metric",
                    value="macro_f1",
                    key="p5_monitor",
                    help="For detection try: metrics/mAP50(B)",
                )

        # ---------------- AUGMENTATION ----------------
        with cfg_aug:
            aug_enabled = st.checkbox("Augmentation enabled", value=True, key="p5_aug_enabled")
            a1, a2 = st.columns(2)
            with a1:
                aug_probability = st.slider("Probability", 0.0, 1.0, 0.5, 0.05, key="p5_aug_prob")
            with a2:
                aug_hflip = st.checkbox("Horizontal flip", value=False, key="p5_aug_flip")

            st.caption("Tick a box to send that value. Unticked = backend default (null).")

            aug_rotation = _opt_num("Rotation degrees", "p5_aug_rot", min_value=0.0, max_value=180.0, value=10.0, step=1.0)
            aug_brightness = _opt_num("Brightness factor", "p5_aug_bri", min_value=0.01, max_value=3.0, value=1.2, step=0.05)
            aug_contrast = _opt_num("Contrast factor", "p5_aug_con", min_value=0.01, max_value=3.0, value=1.2, step=0.05)
            aug_saturation = _opt_num("Saturation factor", "p5_aug_sat", min_value=0.01, max_value=3.0, value=1.2, step=0.05)
            aug_hue = _opt_num("Hue shift", "p5_aug_hue", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
            aug_blur = _opt_num("Blur radius", "p5_aug_blur", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            aug_noise = _opt_num("Noise std", "p5_aug_noise", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
            aug_crop = _opt_num("Crop fraction", "p5_aug_crop", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
            aug_occlusion = _opt_num("Random occlusion fraction", "p5_aug_occ", min_value=0.0, max_value=0.9, value=0.1, step=0.05)

        # ---------------- OPTIMIZER & SCHEDULER ----------------
        with cfg_opt:
            st.markdown("#### Optimizer")
            o1, o2 = st.columns(2)
            with o1:
                optimizer_name = st.selectbox("Optimizer", ["adamw", "adam", "sgd"], key="p5_opt_name")
                learning_rate = st.number_input(
                    "Learning rate", min_value=0.000001, max_value=1.0,
                    value=0.0002, step=0.0001, format="%.6f", key="p5_lr",
                )
            with o2:
                weight_decay = st.number_input(
                    "Weight decay", min_value=0.0, max_value=1.0,
                    value=0.0, step=0.0001, format="%.6f", key="p5_wd",
                )
            momentum = _opt_num("Momentum", "p5_momentum", min_value=0.0, max_value=1.0, value=0.9, step=0.01)

            st.markdown("#### Scheduler")
            scheduler_name = st.selectbox("Scheduler", ["none", "cosine", "step"], key="p5_sched_name")
            warmup_epochs = st.number_input("Warmup epochs", min_value=0, max_value=100, value=0, step=1, key="p5_warmup")
            sched_step_size = _opt_num("Step size", "p5_sched_step", min_value=1, max_value=1000, value=10, step=1)
            sched_gamma = _opt_num("Gamma", "p5_sched_gamma", min_value=0.0001, max_value=1.0, value=0.1, step=0.01)
            sched_min_lr = _opt_num("Min LR", "p5_sched_minlr", min_value=0.0, max_value=1.0, value=0.000001, step=0.000001, format="%.6f")

        # ---------------- LOSS ----------------
        with cfg_loss:
            loss_name = st.selectbox("Loss", ["cross_entropy", "focal"], key="p5_loss_name")
            label_smoothing = st.number_input(
                "Label smoothing", min_value=0.0, max_value=0.5,
                value=0.0, step=0.01, key="p5_label_smooth",
            )
            focal_gamma = _opt_num("Focal gamma", "p5_focal", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
            class_weights_text = st.text_input(
                "Class weights (comma separated, optional)",
                value="",
                key="p5_class_weights",
                help="Example: 1.0, 2.5, 1.0",
            )

        # ---------------- ADVANCED ----------------
        with cfg_adv:
            mixed_precision = st.checkbox("Mixed precision", value=True, key="p5_amp")
            grad_accum = st.number_input("Gradient accumulation steps", min_value=1, max_value=128, value=1, step=1, key="p5_grad_accum")
            max_grad_norm = _opt_num("Max grad norm", "p5_max_grad", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
            early_stop = _opt_num("Early stopping patience", "p5_early", min_value=1, max_value=100, value=5, step=1)
            freeze_mode = st.selectbox("Freeze mode", ["none", "backbone", "last_n_blocks"], key="p5_freeze_mode")
            freeze_last_n = _opt_num("Freeze last N blocks", "p5_freeze_n", min_value=1, max_value=100, value=1, step=1)

        # ---------------- VALIDATION ----------------
        class_weights = None
        class_weights_valid = True
        if class_weights_text.strip():
            try:
                class_weights = parse_numeric_list(class_weights_text)
            except ValueError:
                class_weights_valid = False
                st.error("Class weights must be comma-separated numbers.")

        experiment_config_valid = bool(
            family_id.strip()
            and experiment_name.strip()
            and model_id.strip()
            and dataset_id.strip()
            and class_weights_valid
        )

        if not experiment_config_valid:
            st.info(
                "Fill in the experiment family ID, experiment name, model ID, "
                "and dataset ID to enable the run button."
            )

        # ---------------- BUILD CHANGES (nulls are skipped) ----------------
        change_pairs = [
            ("epochs", int(epochs)),
            ("seed", int(p5_seed)),
            ("monitor", monitor.strip()),
            ("mixed_precision", bool(mixed_precision)),
            ("gradient_accumulation_steps", int(grad_accum)),
            ("data.batch_size", int(p5_batch_size)),
            ("data.input_width", int(p5_input_width)),
            ("data.input_height", int(p5_input_height)),
            ("data.num_workers", int(p5_num_workers)),
            ("data.sampler", sampler),
            ("data.augmentation.enabled", bool(aug_enabled)),
            ("data.augmentation.probability", float(aug_probability)),
            ("data.augmentation.horizontal_flip", bool(aug_hflip)),
            ("optimizer.name", optimizer_name),
            ("optimizer.learning_rate", float(learning_rate)),
            ("optimizer.weight_decay", float(weight_decay)),
            ("scheduler.name", scheduler_name),
            ("scheduler.warmup_epochs", int(warmup_epochs)),
            ("loss.name", loss_name),
            ("loss.label_smoothing", float(label_smoothing)),
            ("freeze.mode", freeze_mode),
        ]

        optional_pairs = [
            ("data.augmentation.rotation_degrees", aug_rotation),
            ("data.augmentation.brightness_factor", aug_brightness),
            ("data.augmentation.contrast_factor", aug_contrast),
            ("data.augmentation.saturation_factor", aug_saturation),
            ("data.augmentation.hue_shift", aug_hue),
            ("data.augmentation.blur_radius", aug_blur),
            ("data.augmentation.noise_std", aug_noise),
            ("data.augmentation.crop_fraction", aug_crop),
            ("data.augmentation.random_occlusion_fraction", aug_occlusion),
            ("optimizer.momentum", momentum),
            ("scheduler.step_size", None if sched_step_size is None else int(sched_step_size)),
            ("scheduler.gamma", sched_gamma),
            ("scheduler.min_lr", sched_min_lr),
            ("loss.focal_gamma", focal_gamma),
            ("loss.class_weights", class_weights),
            ("max_grad_norm", max_grad_norm),
            ("early_stopping_patience", None if early_stop is None else int(early_stop)),
            ("freeze.last_n_blocks", None if freeze_last_n is None else int(freeze_last_n)),
        ]

        for path, value in optional_pairs:
            if value is not None:
                change_pairs.append((path, value))

        specs = []
        if experiment_config_valid:
            specs = [{
                "name": experiment_name.strip(),
                "hypothesis": {
                    "claim": hypothesis_claim.strip(),
                    "expected_direction": "improve",
                },
                "intervention": {
                    "kind": "training",
                    "changes": [
                        {"path": path, "value": value}
                        for path, value in change_pairs
                    ],
                },
                "metrics": ["accuracy", "macro_f1"],
                "n_bootstrap": 1000,
                "n_permutations": 5000,
                "population": {
                    "analysis_id": None,
                    "sample_ids": None,
                    "slice_ids": None,
                },
                "practical_threshold": 0.01,
                "repetitions": 1,
                "seed": int(p5_seed),
                "alpha": float(alpha),
            }]
        with st.expander("Experiment spec preview", expanded=False):
            st.json(specs)

        if st.button(
            "▶ RUN DETECTION EXPERIMENT",
            type="primary",
            use_container_width=True,
            disabled=not experiment_config_valid,
            key="p5_run",
        ):
            with st.spinner("Running detection experiment..."):
                result = api.experiments(
                    family_id=family_id.strip(),
                    baseline_evaluation_id=(
                        baseline_evaluation_id.strip()
                        or None
                    ),
                    specs=specs,
                    matrices=[],
                    model_id=model_id.strip(),
                    dataset_id=dataset_id.strip(),
                    subpath=None,
                    device="auto",
                    force=force,
                    stop_on_failure=stop_on_failure,
                    alpha=float(alpha),
                )

            st.session_state.last_response = result
            st.session_state.last_experiment_response = result
            if result.get("ok"):
                st.success("Detection experiment submitted successfully.")
            else:
                st.error("POST /experiments returned an error. Trying to load the family anyway.")

            show_response(result)

            with st.spinner("Loading experiment family results..."):
                _bundle = _fetch_family_bundle(family_id.strip())

            _render_family_bundle(_bundle)
# ============================================================
# RESULTS
# ============================================================

elif page == "Results":

    st.title("📊 Results")

    st.caption(
        "Inspect the latest backend response and experiment outputs."
    )

    # ---------------- Experiment results ----------------
    st.markdown("## 🧪 Experiment Results")

    lookup_col, btn_col = st.columns([4, 1])
    with lookup_col:
        exp_job_id = st.text_input(
            "Load experiment results from a job ID (optional)",
            key="exp_job_lookup",
        )
    with btn_col:
        st.write("")
        st.write("")
        if st.button("Load", key="exp_job_load") and exp_job_id.strip():
            st.session_state.last_experiment_response = api.job(exp_job_id.strip())
            st.rerun()

    # ---------------- Experiment family (auto-loaded) ----------------
    fam_col, ref_col = st.columns([4, 1])
    with fam_col:
        results_family_id = st.text_input(
            "Family ID",
            value=st.session_state.get("last_family_id", ""),
            key="results_family_id_input",
            help="Filled automatically after you run Phase 5.",
        )
    with ref_col:
        st.write("")
        st.write("")
        refresh = st.button("Load / Refresh", key="family_refresh_btn")

    if refresh and results_family_id.strip():
        with st.spinner("Loading family..."):
            _fetch_family_bundle(results_family_id.strip())

    _bundle = st.session_state.get("experiment_bundle")

    if _bundle:
        _render_family_bundle(_bundle)
    else:
        st.info("Run Phase 5, or type a family ID and click Load / Refresh.")

    st.divider()

    saved_reports = _find_local_failure_analysis_reports()

    if saved_reports:
        saved_options = [
            item["evaluation_id"]
            for item in saved_reports
        ]

        selected_evaluation_id = st.selectbox(
            "Saved evaluation ID",
            options=saved_options,
            index=0,
            help=(
                "Open a saved local failure-analysis report from the "
                "artifacts directory."
            ),
        )

        selected_report = next(
            (
                item["report"]
                for item in saved_reports
                if item["evaluation_id"] == selected_evaluation_id
            ),
            None,
        )

        if selected_report is not None:
            st.markdown("### Saved failure analysis report")
            st.caption(
                "Loaded from the local artifacts directory for the chosen "
                "evaluation ID."
            )
            _render_failure_analysis_report(selected_report)

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

        # Show the complete persisted report for completed failure-analysis jobs.
        if not _failure_analysis_render_job_result(api, response):
            show_response(response)

        saved_report = st.session_state.get(
            "last_analysis_report"
        )

        if isinstance(saved_report, dict):

            _render_failure_analysis_report(
                saved_report
            )

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
                "mean_iou",
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
# Phase 5 detection campaign results
# Environment: Kaggle
# ============================================================

elif page == "Experiments":

    st.title("🧪 Experiments")

    st.caption(
        "Phase 5 controlled experiment families and rankings."
    )

    if not st.session_state.last_response:

        st.info(
            "Run a controlled experiment from "
            "Laboratory → Phase 5 — Controlled Experiments."
        )

    else:

        response = st.session_state.last_response

        if response.get("ok"):
            st.success(
                "Latest experiment family completed."
            )
        else:
            st.error(
                "Latest experiment family returned an error."
            )

        data = extract_data(response)

        if isinstance(data, dict):

            family_id = data.get("family_id")

            if family_id:

                st.markdown(
                    f"### Experiment family: `{family_id}`"
                )

            completed = data.get("completed")
            failed = data.get("failed")

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

            # ------------------------------------------------
            # Ranking
            # ------------------------------------------------

            ranking = data.get("ranking")

            if ranking:

                st.markdown("### Experiment Ranking")

                for row in ranking:

                    rank = row.get("rank", "-")
                    experiment_id = row.get(
                        "experiment_id",
                        "unknown",
                    )
                    score = row.get("score")

                    st.markdown(
                        f"**#{rank} — `{experiment_id}`**"
                    )

                    if score is not None:

                        st.metric(
                            "F1 improvement",
                            f"{float(score):+.6f}",
                        )

                    st.json(row)

            # ------------------------------------------------
            # Outcomes
            # ------------------------------------------------

            outcomes = data.get("outcomes")

            if outcomes:

                st.markdown("### Outcomes")

                st.json(outcomes)

            # ------------------------------------------------
            # Campaign report
            # ------------------------------------------------

            report = data.get("report")

            if report:

                st.markdown("### Campaign Report")

                st.json(report)

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

                        if not _failure_analysis_render_job_result(
                            api,
                            job_response,
                        ):
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