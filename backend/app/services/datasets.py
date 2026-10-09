
import io
import json
import logging
import re
import uuid

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.services.data_loader import load_dataset

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = ROOT / "data" / "uploads"
MAX_UPLOAD_SIZE = 20 * 1024 * 1024


@dataclass
class DatasetContext:
    df: pd.DataFrame
    dataset_id: str
    source: str
    target: str
    positive_class: str
    negative_class: str
    class_selection: str = "automatic"


def parse_csv(content: bytes) -> pd.DataFrame:
    if not content:
        raise ValueError("Le fichier CSV est vide.")

    if len(content) > MAX_UPLOAD_SIZE:
        raise ValueError("Taille maximale : 20 Mo.")

    try:
        df = pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )
    except Exception as exc:
        raise ValueError(
            "Impossible de lire le CSV. Vérifiez son format UTF-8."
        ) from exc

    if df.empty or len(df.columns) < 2:
        raise ValueError(
            "Le CSV doit contenir des observations "
            "et au moins deux colonnes."
        )

    return df


def get_classes(series: pd.Series) -> list[str]:
    if series.isna().any():
        return []

    return sorted(set(series.astype(str).tolist()))


def resolve_positive_class(
    classes: list[str],
    chosen: str | None = None
) -> tuple[str, str]:

    if chosen is not None:
        if chosen not in classes:
            raise ValueError(
                "La classe positive indiquée n'existe pas."
            )
        return chosen, "manual"

    recognized = {
        "yes", "true", "1", "1.0",
        "positive", "left", "exited",
        "churned"
    }

    for cls in classes:
        if cls.strip().lower() in recognized:
            return cls, "automatic"

    # Convention pour les classes non reconnues :
    # dernière valeur dans l'ordre alphabétique.
    return classes[-1], "convention"


def inspect_csv(content: bytes) -> dict:
    df = parse_csv(content)

    candidates = {}

    for column in df.columns:
        classes = get_classes(df[column])

        if len(classes) == 2:
            candidates[column] = classes

    logger.info(
        "CSV inspecté : %d observations, %d colonnes",
        len(df),
        len(df.columns)
    )

    return {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "binary_candidates": candidates
    }


def register_csv(
    content: bytes,
    target: str,
    positive_class: str | None = None
) -> dict:

    df = parse_csv(content)

    if not target or target not in df.columns:
        raise ValueError(
            "Vous devez indiquer une variable cible valide."
        )

    classes = get_classes(df[target])

    if len(classes) != 2:
        raise ValueError(
            "La variable cible doit contenir exactement "
            "deux classes et aucune valeur manquante."
        )

    positive, selection = resolve_positive_class(
        classes,
        positive_class
    )

    negative = next(
        cls for cls in classes if cls != positive
    )

    dataset_id = uuid.uuid4().hex
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = UPLOAD_DIR / f"{dataset_id}.csv"
    meta_path = UPLOAD_DIR / f"{dataset_id}.json"

    df.to_csv(csv_path, index=False)

    metadata = {
        "target": target,
        "positive_class": positive,
        "negative_class": negative,
        "class_selection": selection
    }

    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8"
    )

    logger.info(
        "Dataset importé : id=%s, lignes=%d, cible=%s, positive=%s",
        dataset_id,
        len(df),
        target,
        positive
    )

    return {
        "dataset_id": dataset_id,
        "source": "csv",
        "rows": len(df),
        "columns_count": len(df.columns),
        **metadata
    }


@lru_cache(maxsize=16)
def load_uploaded_dataset(dataset_id: str) -> DatasetContext:

    if not re.fullmatch(r"[a-f0-9]{32}", dataset_id):
        raise ValueError("Identifiant invalide.")

    csv_path = UPLOAD_DIR / f"{dataset_id}.csv"
    meta_path = UPLOAD_DIR / f"{dataset_id}.json"

    if not csv_path.exists() or not meta_path.exists():
        raise ValueError("Dataset introuvable.")

    metadata = json.loads(
        meta_path.read_text(encoding="utf-8")
    )

    return DatasetContext(
        df=pd.read_csv(csv_path),
        dataset_id=dataset_id,
        source="csv",
        **metadata
    )


def get_dataset(
    dataset_id: str | None = None
) -> DatasetContext:

    if dataset_id and dataset_id != "kaggle":
        return load_uploaded_dataset(dataset_id)

    return DatasetContext(
        df=load_dataset(),
        dataset_id="kaggle",
        source="kaggle",
        target="Attrition",
        positive_class="Yes",
        negative_class="No"
    )
