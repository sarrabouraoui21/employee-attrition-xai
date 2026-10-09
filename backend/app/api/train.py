import logging

from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException
)

from app.schemas.training import TrainingRequest
from app.services.training_jobs import (
    create_job,
    get_job,
    run_training_job
)

router = APIRouter(
    prefix="/train",
    tags=["Entraînement"]
)

logger = logging.getLogger(__name__)


@router.post("/start")
def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    try:
        job_id = create_job(request)

        background_tasks.add_task(
            run_training_job,
            job_id,
            request
        )

        logger.info(
            "POST /train/start | job=%s",
            job_id
        )

        return {
            "job_id": job_id,
            "state": "queued",
            "message": "Entraînement lancé."
        }

    except ValueError as error:
        logger.warning(
            "Entraînement refusé : %s", error
        )

        raise HTTPException(
            status_code=409,
            detail=str(error)
        ) from error


@router.get("/status/{job_id}")
def training_status(job_id: str):
    job = get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Entraînement introuvable."
        )

    return job