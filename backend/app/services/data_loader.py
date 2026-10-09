
import logging
import os
from functools import lru_cache
from pathlib import Path

import kagglehub
import pandas as pd

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    dataset_id = os.getenv(
        "KAGGLE_DATASET",
        "pavansubhasht/ibm-hr-analytics-attrition-dataset"
    )

    try:
        logger.info("Téléchargement du dataset : %s", dataset_id)

        path = Path(kagglehub.dataset_download(dataset_id))
        csv_files = list(
            path.rglob("WA_Fn-UseC_-HR-Employee-Attrition.csv")
        )

        if not csv_files:
            raise FileNotFoundError("Fichier CSV introuvable")

        df = pd.read_csv(csv_files[0])

        if "Attrition" not in df.columns:
            raise ValueError("Variable cible Attrition absente")

        logger.info(
            "Dataset chargé : %s lignes, %s colonnes",
            df.shape[0],
            df.shape[1]
        )

        return df

    except Exception:
        logger.exception("Erreur lors du chargement du dataset")
        raise
