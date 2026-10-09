from typing import Literal

from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    dataset_id: str | None = None

    features: list[str] | None = None

    test_size: float = Field(
        default=0.20,
        ge=0.10,
        le=0.40
    )

    random_state: int = Field(default=42, ge=0)

    numeric_imputation: Literal[
        "median", "mean"
    ] = "median"

    scale_numeric: bool = True

    inspect_numeric: str | None = None
    inspect_categorical: str | None = None