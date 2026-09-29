from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvaluationResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    dataset_id: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    prediction_artifact: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    configuration: dict[str, Any] = Field(default_factory=dict)
