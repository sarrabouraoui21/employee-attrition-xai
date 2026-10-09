import copy
import logging
import threading
import uuid

from app.schemas.training import TrainingRequest
from app.services.training import train_models

logger = logging.getLogger(__name__)

_jobs = {}
_lock = threading.Lock()


def create_job(request: TrainingRequest) -> str:
    with _lock:
        active = any(
            item["state"] in {"queued", "running"}
            for item in _jobs.values()
        )

        if active:
            raise ValueError(
                "Un entraînement est déjà en cours. "
                "Attendez sa fin avant d'en lancer un autre."
            )

        job_id = uuid.uuid4().hex

        _jobs[job_id] = {
            "job_id": job_id,
            "dataset_id": (
                request.preprocessing.dataset_id or "kaggle"
            ),
            "state": "queued",
            "stage": "En attente de démarrage",
            "progress": 0,
            "total_models": len(request.models),
            "results": [],
            "error": None
        }

    logger.info(
        "JOB CREATED | id=%s | models=%d",
        job_id, len(request.models)
    )

    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return copy.deepcopy(job) if job else None


def update_job(
    job_id: str,
    stage=None,
    progress=None,
    result=None,
    state=None,
    error=None
):
    with _lock:
        job = _jobs[job_id]

        if stage is not None:
            job["stage"] = stage

        if progress is not None:
            job["progress"] = min(
                100, max(0, int(progress))
            )

        if result is not None:
            job["results"].append(result)

        if state is not None:
            job["state"] = state

        if error is not None:
            job["error"] = error


def run_training_job(
    job_id: str,
    request: TrainingRequest
):
    update_job(
        job_id,
        state="running",
        stage="Démarrage de l'entraînement"
    )

    try:
        train_models(
            request=request,
            job_id=job_id,
            notify=lambda **kwargs: update_job(
                job_id, **kwargs
            )
        )

        update_job(
            job_id,
            state="completed",
            stage="Tous les modèles sont entraînés",
            progress=100
        )

        logger.info("JOB SUCCESS | id=%s", job_id)

    except ValueError as error:
        logger.exception(
            "JOB VALIDATION ERROR | id=%s", job_id
        )

        update_job(
            job_id,
            state="failed",
            stage="Entraînement interrompu",
            error=str(error)
        )

    except Exception:
        logger.exception(
            "JOB FAILED | id=%s", job_id
        )

        update_job(
            job_id,
            state="failed",
            stage="Erreur pendant l'entraînement",
            error="Erreur serveur. Consultez les logs FastAPI."
        )