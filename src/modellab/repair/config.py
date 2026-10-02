from pydantic import BaseModel, ConfigDict, Field


class RepairCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_target_improvement: float = Field(0.05, ge=0.0, le=1.0)
    require_significance: bool = True
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    max_global_drop: float = Field(0.005, ge=0.0, le=1.0)
    max_new_error_rate: float = Field(0.01, ge=0.0, le=1.0)
    max_class_recall_drop: float = Field(0.02, ge=0.0, le=1.0)
    max_slice_regression: float = Field(0.02, ge=0.0, le=1.0)
    min_support: int = Field(10, ge=1)


class RepairConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criteria: RepairCriteria = Field(default_factory=RepairCriteria)
    weight_boost: tuple[float, ...] = (1.5, 2.0, 3.0)
    weight_reduce: tuple[float, ...] = (0.8, 0.6, 0.5)
    thresholds: tuple[float, ...] = (0.6, 0.7, 0.8)
    factors_low: tuple[float, ...] = (1.3, 1.5, 2.0)
    factors_high: tuple[float, ...] = (0.8, 0.6)
    important_slices: int = Field(5, ge=0)
    seed: int = Field(42, ge=0, le=2**31)
    n_bootstrap: int = Field(300, ge=100)
    n_permutations: int = Field(2000, ge=100)
