import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Experiment(BaseModel):
    experiment_id: str = Field(default_factory=_new_id)
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    seed: int = Field(ge=0, lt=2**32)
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.CREATED
