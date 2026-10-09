
import json
import logging
import re

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split

from app.schemas.preprocessing import PreprocessingConfig
from app.services.datasets import get_dataset
from app.services.training_jobs import get_job

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


# =====================================================
# 1. ENTRAÎNEMENTS DISPONIBLES
# =====================================================

def list_training_runs() -> list:
    jobs = {}

    if not ARTIFACT_DIR.exists():
        return []

    for path in ARTIFACT_DIR.glob("*.joblib"):
        match = re.fullmatch(
            r"([a-f0-9]{32})_(.+)",
            path.stem
        )

        if not match:
            continue

        job_id, model = match.groups()

        if model not in MODEL_NAMES:
            continue

        if job_id not in jobs:
            jobs[job_id] = {
                "job_id": job_id,
                "models": [],
                "modified_at": 0
            }

        jobs[job_id]["models"].append(model)
        jobs[job_id]["modified_at"] = max(
            jobs[job_id]["modified_at"],
            path.stat().st_mtime
        )

    return sorted(
        jobs.values(),
        key=lambda item: item["modified_at"],
        reverse=True
    )


# =====================================================
# 2. MÉTRIQUES
# =====================================================

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
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
            float(roc_auc_score(
                y_true, probabilities
            )), 4
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
        },
        "predicted_positives": int(
            predictions.sum()
        )
    }


def sample_curve(x, y, max_points=300):
    indexes = np.linspace(
        0,
        len(x) - 1,
        min(len(x), max_points),
        dtype=int
    )

    indexes = np.unique(indexes)

    return [
        {
            "x": round(float(x[i]), 6),
            "y": round(float(y[i]), 6)
        }
        for i in indexes
    ]


# =====================================================
# 3. RECONSTRUCTION DU MÊME JEU DE TEST
# =====================================================

@lru_cache(maxsize=8)
def load_predictions(job_id: str, config_json: str):
    config = PreprocessingConfig.model_validate_json(
        config_json
    )

    logger.info(
        "EVALUATION | Loading models | job=%s",
        job_id
    )

    current_job = get_job(job_id)

    if current_job is not None:
        if current_job["state"] != "completed":
            raise ValueError(
                "Cet entraînement n'est pas encore terminé."
            )

    context = get_dataset(config.dataset_id)

    y = (
        context.df[context.target]
        .astype(str)
        .eq(context.positive_class)
        .astype(int)
    )

    if y.nunique() != 2:
        raise ValueError(
            "La cible doit contenir deux classes."
        )

    # Retrouver exactement les mêmes indices de test
    # grâce au split stratifié et au random_state.
    indices = np.arange(len(context.df))

    _, test_indices = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )

    y_test = y.iloc[test_indices].to_numpy()

    predictions = {}
    metadata = {}

    for model_name, display_name in MODEL_NAMES.items():
        artifact_path = (
            ARTIFACT_DIR
            / f"{job_id}_{model_name}.joblib"
        )

        if not artifact_path.is_file():
            continue

        logger.info(
            "EVALUATION | Loading %s",
            model_name
        )

        # Les fichiers joblib doivent provenir
        # exclusivement de notre backend.
        artifact = joblib.load(artifact_path)

        if (
            artifact["dataset_id"] != context.dataset_id
            or artifact["target"] != context.target
            or artifact["positive_class"]
            != context.positive_class
        ):
            raise ValueError(
                f"Métadonnées incompatibles : {model_name}"
            )

        features = artifact["features"]

        if config.features is not None:
            if list(config.features) != list(features):
                raise ValueError(
                    "Les variables ont changé depuis "
                    "l'entraînement. Utilisez la configuration "
                    "initiale ou réentraînez les modèles."
                )

        X_test = (
            context.df[features]
            .iloc[test_indices]
            .copy()
        )

        pipeline = artifact["pipeline"]

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        calculated = calculate_metrics(
            y_test,
            probabilities,
            0.5
        )

        # Vérification du même jeu de test.
        saved = artifact["metrics"]

        for key in ["precision", "recall", "f1",
                    "roc_auc", "pr_auc"]:
            if abs(calculated[key] - saved[key]) > 0.00015:
                raise ValueError(
                    "Le jeu de test ne correspond pas "
                    "à celui de l'entraînement. Vérifiez "
                    "test_size, random_state et les données."
                )

        predictions[model_name] = probabilities

        metadata[model_name] = {
            "display_name": display_name,
            "artifact": artifact_path.name
        }

    if not predictions:
        raise ValueError(
            "Aucun modèle entraîné trouvé pour ce job."
        )

    logger.info(
        "EVALUATION READY | models=%d | test=%d",
        len(predictions),
        len(y_test)
    )

    return {
        "y_test": y_test,
        "predictions": predictions,
        "metadata": metadata,
        "target": context.target,
        "positive_class": context.positive_class,
        "negative_class": context.negative_class
    }


def get_prediction_data(job_id, preprocessing):
    config_json = preprocessing.model_dump_json(
        exclude={
            "inspect_numeric",
            "inspect_categorical"
        }
    )

    return load_predictions(job_id, config_json)


# =====================================================
# 4. RAPPORT COMPARATIF
# =====================================================

def build_evaluation_report(job_id, preprocessing):
    data = get_prediction_data(
        job_id,
        preprocessing
    )

    y_test = data["y_test"]
    results = []

    for name, probabilities in data[
        "predictions"
    ].items():

        metrics = calculate_metrics(
            y_test, probabilities, 0.5
        )

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities
        )

        precision, recall, _ = (
            precision_recall_curve(
                y_test,
                probabilities
            )
        )

        results.append({
            "model": name,
            "display_name": (
                data["metadata"][name]["display_name"]
            ),
            "metrics": metrics,
            "roc_curve": sample_curve(fpr, tpr),
            "pr_curve": sample_curve(
                recall, precision
            )
        })

        logger.info(
            "EVALUATION | %s | Recall=%.4f | AP=%.4f",
            name,
            metrics["recall"],
            metrics["pr_auc"]
        )

    return {
        "job_id": job_id,
        "target": data["target"],
        "positive_class": data["positive_class"],
        "negative_class": data["negative_class"],
        "test_size": len(y_test),
        "positive_count": int(y_test.sum()),
        "positive_rate": round(
            float(y_test.mean()), 4
        ),
        "results": results
    }


# =====================================================
# 5. ÉVALUATION AVEC SEUIL PERSONNALISÉ
# =====================================================

def evaluate_threshold(
    job_id,
    preprocessing,
    model,
    threshold
):
    data = get_prediction_data(
        job_id,
        preprocessing
    )

    if model not in data["predictions"]:
        raise ValueError(
            "Ce modèle n'appartient pas à cet entraînement."
        )

    metrics = calculate_metrics(
        data["y_test"],
        data["predictions"][model],
        threshold
    )

    cm = metrics["confusion_matrix"]

    logger.info(
        "THRESHOLD | model=%s | threshold=%.2f "
        "| TP=%d FP=%d FN=%d TN=%d",
        model,
        threshold,
        cm["tp"],
        cm["fp"],
        cm["fn"],
        cm["tn"]
    )

    return {
        "job_id": job_id,
        "model": model,
        "threshold": threshold,
        "metrics": metrics,
        "test_size": len(data["y_test"]),
        "positive_class": data["positive_class"]
    }
