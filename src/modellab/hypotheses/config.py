from pydantic import BaseModel, ConfigDict, Field


class HypothesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_failures: int = Field(3, ge=1)
    max_slice_conditions: int = Field(2, ge=1, le=3)
    overlap_skip: float = Field(0.5, gt=0.0, le=1.0)
    min_failure_errors: int = Field(5, ge=1)
    include_classes: bool = True
    include_confusions: bool = True
    min_confusion_count: int = Field(10, ge=2)

    remedy_factor_low: float = Field(1.5, gt=1.0)
    remedy_factor_high: float = Field(0.7, gt=0.0, lt=1.0)
    overpredict_weight: float = Field(0.5, gt=0.0, lt=1.0)
    bias_boost_weight: float = Field(2.0, gt=1.0)
    induce_blur_radius: float = Field(2.0, gt=0.0)
    induce_jpeg_quality: int = Field(20, ge=1, le=95)
    control_size: int = Field(60, ge=10)

    dominant_prediction_share: float = Field(0.6, gt=0.0, le=1.0)
    ambiguous_share: float = Field(0.5, gt=0.0, le=1.0)
    high_confidence_share: float = Field(0.5, gt=0.0, le=1.0)

    max_experiments: int = Field(6, ge=1)
    seed: int = Field(42, ge=0, le=2**31)
    practical_threshold: float = Field(0.01, ge=0.0, le=1.0)
    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    allow_image_experiments: bool = True
