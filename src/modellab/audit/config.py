from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layout: Literal["auto", "class_only", "split_class"] = "auto"
    class_names: tuple[str, ...] | None = None

    path_column: str = "path"
    label_column: str = "label"
    label_name_column: str | None = None
    split_column: str | None = "split"
    id_column: str | None = None

    near_duplicate_threshold: int = Field(4, ge=0, le=16)
    max_bucket_size: int = Field(5000, ge=2)

    min_side: int = Field(32, ge=1)
    max_side: int = Field(10000, ge=1)
    max_aspect_ratio: float = Field(4.0, gt=1.0)
    min_file_bytes: int = Field(256, ge=0)
    max_file_bytes: int = Field(50 * 1024 * 1024, ge=1)

    imbalance_ratio: float = Field(5.0, gt=1.0)
    severe_imbalance_ratio: float = Field(20.0, gt=1.0)
    split_proportion_gap: float = Field(0.15, gt=0.0, le=1.0)
    min_split_size_for_gap: int = Field(20, ge=1)

    outlier_z: float = Field(4.0, gt=0.0)
    min_samples_for_outliers: int = Field(20, ge=3)
    compute_stats: bool = True
    min_contrast: float = Field(0.01, ge=0.0, le=1.0)

    filename_overlap_min_stem: int = Field(6, ge=1)
    max_evidence: int = Field(20, ge=1)
    max_listed_groups: int = Field(200, ge=1)

    @model_validator(mode="after")
    def _check(self):
        if self.severe_imbalance_ratio < self.imbalance_ratio:
            raise ValueError("severe_imbalance_ratio must not be below imbalance_ratio")
        if self.min_side > self.max_side:
            raise ValueError("min_side must not exceed max_side")
        if self.min_file_bytes > self.max_file_bytes:
            raise ValueError("min_file_bytes must not exceed max_file_bytes")
        return self
