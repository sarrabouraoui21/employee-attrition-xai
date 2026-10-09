
import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from xgboost import XGBClassifier

from app.schemas.training import TrainingRequest
from app.services.datasets import get_dataset
from app.services.preprocessing import prepare_dataset

logger = logging.getLogger(__name__)

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts"
)

MODEL_NAMES = {
    "logistic_regression": "Régression logistique",
    "decision_tree": "Arbre de décision",
    "knn": "KNN",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost"
}


def build_model(name, hp, seed, class_ratio):
    if name == "logistic_regression":
        model = LogisticRegression(
            C=hp.logistic_c,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed
        )
        params = {
            "model__C": [
                hp.logistic_c * 0.5,
                hp.logistic_c,
                hp.logistic_c * 2
            ]
        }

    elif name == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=hp.tree_max_depth,
            min_samples_leaf=hp.tree_min_samples_leaf,
            class_weight="balanced",
            random_state=seed
        )
        params = {
            "model__max_depth": [
                max(2, hp.tree_max_depth - 2),
                hp.tree_max_depth,
                hp.tree_max_depth + 2
            ],
            "model__min_samples_leaf": [
                hp.tree_min_samples_leaf,
                hp.tree_min_samples_leaf + 2
            ]
        }

    elif name == "knn":
        model = KNeighborsClassifier(
            n_neighbors=hp.knn_neighbors,
            weights="distance"
        )
        params = {
            "model__n_neighbors": [
                max(1, hp.knn_neighbors - 2),
                hp.knn_neighbors,
                hp.knn_neighbors + 2
            ],
            "model__weights": ["uniform", "distance"]
        }

    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=hp.rf_n_estimators,
            max_depth=hp.rf_max_depth,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1
        )
        params = {
            "model__n_estimators": [
                max(20, hp.rf_n_estimators // 2),
                hp.rf_n_estimators
            ],
            "model__max_depth": [
                max(2, hp.rf_max_depth - 3),
                hp.rf_max_depth
            ],
            "model__min_samples_leaf": [1, 2, 4]
        }

    elif name == "xgboost":
        model = XGBClassifier(
            n_estimators=hp.xgb_n_estimators,
            max_depth=hp.xgb_max_depth,
            learning_rate=hp.xgb_learning_rate,
            scale_pos_weight=class_ratio,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1
        )
        params = {
            "model__n_estimators": [
                max(20, hp.xgb_n_estimators // 2),
                hp.xgb_n_estimators
            ],
            "model__max_depth": [
                max(2, hp.xgb_max_depth - 1),
                hp.xgb_max_depth
            ],
            "model__learning_rate": [
                max(0.01, hp.xgb_learning_rate / 2),
                hp.xgb_learning_rate
            ]
        }

    else:
        raise ValueError(f"Modèle inconnu : {name}")

    return model, params


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": round(
            float(accuracy_score(y_true, predictions)), 4
        ),
        "precision": round(
            float(precision_score(
                y_true, predictions, zero_division=0
            )), 4
        ),
        "recall": round(
            float(recall_score(
                y_true, predictions, zero_division=0
            )), 4
        ),
        "f1": round(
            float(f1_score(
                y_true, predictions, zero_division=0
            )), 4
        ),
        "roc_auc": round(
            float(roc_auc_score(y_true, probabilities)), 4
        ),
        "pr_auc": round(
            float(average_precision_score(
                y_true, probabilities
            )), 4
        ),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }


def train_models(request: TrainingRequest, job_id: str, notify):
    logger.info(
        "TRAINING START | job=%s | models=%s",
        job_id, request.models
    )

    if len(set(request.models)) != len(request.models):
        raise ValueError("Liste de modèles dupliquée.")

    if not request.models:
        raise ValueError("Aucun modèle sélectionné.")

    notify(stage="Chargement et préparation", progress=2)

    context = get_dataset(
        request.preprocessing.dataset_id
    )

    prepared = prepare_dataset(
        context, request.preprocessing
    )

    X_train = prepared.X_train_raw
    X_test = prepared.X_test_raw
    y_train = prepared.y_train
    y_test = prepared.y_test

    if y_test.nunique() != 2:
        raise ValueError(
            "Le jeu de test doit contenir les deux classes."
        )

    minority_count = int(y_train.value_counts().min())

    if (
        request.tune_hyperparameters
        and minority_count < request.cv_folds
    ):
        raise ValueError(
            "Pas assez d'observations minoritaires "
            "pour cette validation croisée."
        )

    if len(X_train) < request.cv_folds * 3:
        raise ValueError(
            "Jeu d'entraînement insuffisant pour la CV."
        )

    cv = StratifiedKFold(
        n_splits=request.cv_folds,
        shuffle=True,
        random_state=request.preprocessing.random_state
    )

    count_positive = int((y_train == 1).sum())
    count_negative = int((y_train == 0).sum())
    class_ratio = count_negative / max(count_positive, 1)

    ARTIFACT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    notify(stage="Préparation terminée", progress=5)

    total = len(request.models)

    for index, name in enumerate(request.models):
        start = time.perf_counter()
        display_name = MODEL_NAMES[name]

        base_progress = 5 + int(90 * index / total)

        logger.info(
            "MODEL %d/%d START | %s",
            index + 1, total, display_name
        )

        notify(
            stage=f"{display_name} : configuration",
            progress=base_progress
        )

        estimator, search_space = build_model(
            name,
            request.hyperparameters,
            request.preprocessing.random_state,
            class_ratio
        )

        # Le pipeline complet entre dans la validation
        # croisée : aucun preprocessing ajusté avant CV.
        pipeline = Pipeline([
            ("preprocessor", clone(prepared.preprocessor)),
            ("model", estimator)
        ])

        # Éviter un nombre de voisins supérieur
        # à la taille d'un fold d'entraînement.
        if name == "knn":
            max_neighbors = max(
                1,
                len(X_train)
                - (len(X_train) + request.cv_folds - 1)
                // request.cv_folds
            )

            pipeline.set_params(
                model__n_neighbors=min(
                    request.hyperparameters.knn_neighbors,
                    max_neighbors
                )
            )

            search_space["model__n_neighbors"] = sorted(
                set(
                    min(value, max_neighbors)
                    for value in
                    search_space["model__n_neighbors"]
                )
            )

        best_cv_score = None
        best_params = {}

        if request.tune_hyperparameters:
            logger.info(
                "MODEL %s | RandomizedSearchCV | folds=%d | iterations=%d",
                name, request.cv_folds,
                request.search_iterations
            )

            notify(
                stage=f"{display_name} : optimisation CV",
                progress=base_progress + 3
            )

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=search_space,
                n_iter=request.search_iterations,
                scoring=request.scoring,
                cv=cv,
                random_state=request.preprocessing.random_state,
                n_jobs=1,
                refit=True,
                error_score="raise"
            )

            search.fit(X_train, y_train)

            trained_pipeline = search.best_estimator_

            best_cv_score = round(
                float(search.best_score_), 4
            )

            best_params = {
                key.replace("model__", ""): (
                    value.item()
                    if hasattr(value, "item")
                    else value
                )
                for key, value in search.best_params_.items()
            }

            logger.info(
                "MODEL %s | Best CV score=%.4f | params=%s",
                name, best_cv_score, best_params
            )
        else:
            notify(
                stage=f"{display_name} : entraînement",
                progress=base_progress + 6
            )

            trained_pipeline = pipeline.fit(
                X_train, y_train
            )

        notify(
            stage=f"{display_name} : évaluation",
            progress=base_progress + 11
        )

        probabilities = trained_pipeline.predict_proba(
            X_test
        )[:, 1]

        metrics = calculate_metrics(
            y_test, probabilities
        )

        artifact_path = (
            ARTIFACT_DIR / f"{job_id}_{name}.joblib"
        )

        joblib.dump(
            {
                "pipeline": trained_pipeline,
                "model_name": name,
                "dataset_id": context.dataset_id,
                "target": context.target,
                "positive_class": context.positive_class,
                "negative_class": context.negative_class,
                "features": X_train.columns.tolist(),
                "metrics": metrics
            },
            artifact_path
        )

        duration = round(
            time.perf_counter() - start, 2
        )

        result = {
            "model": name,
            "display_name": display_name,
            "metrics": metrics,
            "cv_score": best_cv_score,
            "best_params": best_params,
            "duration_seconds": duration,
            "artifact": artifact_path.name
        }

        logger.info(
            "MODEL %s DONE | %.2fs | recall=%.4f | PR-AUC=%.4f",
            name, duration,
            metrics["recall"],
            metrics["pr_auc"]
        )

        notify(
            stage=f"{display_name} : terminé",
            progress=5 + int(90 * (index + 1) / total),
            result=result
        )

    logger.info(
        "TRAINING COMPLETE | job=%s | count=%d",
        job_id, total
    )
