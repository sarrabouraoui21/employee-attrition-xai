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

ALL_MODELS = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "random_forest",
    "xgboost"
]


class Hyperparameters(BaseModel):
    logistic_c: float = Field(1.0, gt=0, le=100)

    tree_max_depth: int = Field(6, ge=2, le=30)
    tree_min_samples_leaf: int = Field(2, ge=1, le=30)

    knn_neighbors: int = Field(9, ge=1, le=100)

    rf_n_estimators: int = Field(120, ge=10, le=500)
    rf_max_depth: int = Field(10, ge=2, le=30)

    xgb_n_estimators: int = Field(100, ge=10, le=500)
    xgb_max_depth: int = Field(4, ge=2, le=15)
    xgb_learning_rate: float = Field(
        0.1, gt=0, le=1
    )


class TrainingRequest(BaseModel):
    preprocessing: PreprocessingConfig

    models: list[ModelName] = Field(
        default_factory=lambda: list(ALL_MODELS),
        min_length=1
    )

    tune_hyperparameters: bool = True

    cv_folds: int = Field(3, ge=2, le=5)
    search_iterations: int = Field(3, ge=1, le=10)

    scoring: Literal[
        "average_precision",
        "f1",
        "recall",
        "roc_auc"
    ] = "average_precision"

    hyperparameters: Hyperparameters = Field(
        default_factory=Hyperparameters
    )