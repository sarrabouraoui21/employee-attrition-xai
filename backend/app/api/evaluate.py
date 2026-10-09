
import logging

from fastapi import APIRouter, HTTPException

from app.schemas.evaluation import (
    EvaluationRequest,
    ThresholdRequest
)

from app.services.evaluation import (
    list_training_runs,
    build_evaluation_report,
    evaluate_threshold
)

router = APIRouter(
    prefix="/evaluate",
    tags=["Évaluation"]
)

logger = logging.getLogger(__name__)


@router.get("/jobs")
def available_jobs():
    logger.info("GET /evaluate/jobs")

    return {
        "jobs": list_training_runs()
    }


@router.post("/report")
def evaluation_report(request: EvaluationRequest):
    logger.info(
        "POST /evaluate/report | job=%s",
        request.job_id
    )

    try:
        return build_evaluation_report(
            request.job_id,
            request.preprocessing
        )

    except ValueError as error:
        logger.warning(
            "Évaluation refusée : %s", error
        )
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception:
        logger.exception("Erreur rapport d'évaluation")
        raise HTTPException(
            status_code=500,
            detail="Erreur serveur. Consulter les logs."
        )


@router.post("/threshold")
def threshold_evaluation(request: ThresholdRequest):
    logger.info(
        "POST /evaluate/threshold | model=%s | threshold=%.2f",
        request.model,
        request.threshold
    )

    try:
        return evaluate_threshold(
            request.job_id,
            request.preprocessing,
            request.model,
            request.threshold
        )

    except ValueError as error:
        logger.warning(
            "Seuil invalide : %s", error
        )
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception:
        logger.exception(
            "Erreur évaluation du seuil"
        )
        raise HTTPException(
            status_code=500,
            detail="Erreur serveur. Consulter les logs."
        )
