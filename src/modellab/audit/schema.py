from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


SEVERITY_RANK = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}


class Category(str, Enum):
    INVENTORY = "inventory"
    LABEL_INTEGRITY = "label_integrity"
    IMAGE_INTEGRITY = "image_integrity"
    EXACT_DUPLICATES = "exact_duplicates"
    NEAR_DUPLICATES = "near_duplicates"
    DISTRIBUTION = "distribution"
    LEAKAGE = "leakage"
    STATISTICS = "statistics"


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    category: Category
    code: str
    severity: Severity
    title: str
    description: str
    count: int = Field(ge=0)
    sample_ids: list[str] = Field(default_factory=list)
    evidence_truncated: bool = False
    affected_splits: list[str] = Field(default_factory=list)
    affected_classes: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class AuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    dataset_id: str
    fingerprint: str
    config: dict[str, Any]
    scan: dict[str, Any]
    inventory: dict[str, Any]
    label_integrity: dict[str, Any]
    image_integrity: dict[str, Any]
    exact_duplicates: dict[str, Any]
    near_duplicates: dict[str, Any]
    distribution: dict[str, Any]
    leakage: dict[str, Any]
    statistics: dict[str, Any]
    findings: list[Finding]
    summary: dict[str, Any]
