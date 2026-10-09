import logging

from fastapi import APIRouter, HTTPException

from app.schemas.preprocessing import PreprocessingConfig
from app.services.datasets import get_dataset
from app.services.preprocessing import (
    build_preprocessing_report,
    get_preprocessing_options
)

router = APIRouter(
    prefix="/preprocessing",
    tags=["Préparation"]
)

logger = logging.getLogger(__name__)


@router.get("/options")
def preprocessing_options(
    dataset_id: str | None = None
):
    logger.info(
        "GET /preprocessing/options | dataset=%s",
        dataset_id or "kaggle"
    )

    try:
        context = get_dataset(dataset_id)
        return get_preprocessing_options(context)

    except ValueError as exc:
        logger.warning("Configuration invalide : %s", exc)
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except Exception:
        logger.exception("Erreur options preprocessing")
        raise HTTPException(
            status_code=500,
            detail="Erreur serveur. Consulter les logs."
        )


@router.post("/preview")
def preprocessing_preview(
    config: PreprocessingConfig
):
    logger.info(
        "POST /preprocessing/preview | dataset=%s",
        config.dataset_id or "kaggle"
    )

    try:
        context = get_dataset(config.dataset_id)

        return build_preprocessing_report(
            context,
            config
        )

    except ValueError as exc:
        logger.warning(
            "Preprocessing refusé : %s",
            exc
        )
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except Exception:
        logger.exception(
            "Erreur pendant le preprocessing"
        )
        raise HTTPException(
            status_code=500,
            detail="Erreur preprocessing. Consulter les logs."
        )