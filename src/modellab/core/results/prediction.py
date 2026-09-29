from pydantic import BaseModel, ConfigDict, Field, model_validator


class Prediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    predicted_class: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    class_probabilities: list[float] | None = None

    @model_validator(mode="after")
    def _check_class_in_range(self):
        probs = self.class_probabilities
        if probs is not None and self.predicted_class >= len(probs):
            raise ValueError("predicted_class is outside class_probabilities")
        return self
