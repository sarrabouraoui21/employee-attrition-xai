
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.preprocessing import PreprocessingConfig


ModelName = Literal[
    "logistic_regression",
    "decision_tree",
    "knn",
    "random_forest",
    "xgboost"
]


class EvaluationRequest(BaseModel):
    job_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    preprocessing: PreprocessingConfig


class ThresholdRequest(EvaluationRequest):
    model: ModelName
    threshold: float = Field(
        default=0.5,
        ge=0.01,
        le=0.99
    )
