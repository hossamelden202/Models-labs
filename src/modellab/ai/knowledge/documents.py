from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_type: Literal["experiment", "evaluation", "failure_analysis"]
    source_id: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
