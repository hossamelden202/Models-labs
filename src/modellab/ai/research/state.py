from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    request: str
    family_id: str
    baseline: dict[str, Any]
    experiments: list[dict[str, Any]]
    failure_source: str
    failure_per_class: dict[str, Any]
    menu: list[dict[str, Any]]
    analysis: dict[str, Any]
    retrieved: list[dict[str, Any]]
    ranking: list[dict[str, Any]]
    choice: dict[str, Any]
    llm_used: bool
    hypothesis: dict[str, Any]
    proposal: dict[str, Any]
    validation: dict[str, Any]
    notes: list[str]
    trace: list[str]
