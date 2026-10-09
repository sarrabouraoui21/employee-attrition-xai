
import json
import logging

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile
)

from app.services.datasets import (
    MAX_UPLOAD_SIZE,
    get_dataset,
    inspect_csv,
    register_csv
)

from app.services.exploration import (
    get_overview,
    get_feature_analysis,
    get_correlations
)

router = APIRouter(prefix="/data", tags=["Données"])
logger = logging.getLogger(__name__)


async def read_upload(file: UploadFile) -> bytes:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise ValueError("Fichier CSV requis.")

    content = await file.read(MAX_UPLOAD_SIZE + 1)

    if len(content) > MAX_UPLOAD_SIZE:
        raise ValueError("Taille maximale : 20 Mo.")

    return content


def load_context(dataset_id: str | None):
    try:
        return get_dataset(dataset_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc


@router.post("/inspect-csv")
async def inspect_uploaded_csv(file: UploadFile = File(...)):
    logger.info("POST /data/inspect-csv")

    try:
        return inspect_csv(await read_upload(file))
    except ValueError as exc:
        logger.warning("CSV invalide : %s", exc)
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc
    except Exception:
        logger.exception("Erreur inspection CSV")
        raise HTTPException(
            status_code=500,
            detail="Erreur serveur. Consulter les logs."
        )


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target: str = Form(...),
    positive_class: str | None = Form(None)
):
    logger.info("POST /data/upload - target=%s", target)

    try:
        result = register_csv(
            content=await read_upload(file),
            target=target,
            positive_class=positive_class
        )

        logger.info(
            "Import réussi : %s",
            result["dataset_id"]
        )

        return result

    except ValueError as exc:
        logger.warning("Import refusé : %s", exc)
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc
    except Exception:
        logger.exception("Erreur import CSV")
        raise HTTPException(
            status_code=500,
            detail="Erreur serveur. Consulter les logs."
        )


@router.get("/overview")
def overview(dataset_id: str | None = None):
    logger.info("GET /data/overview - id=%s", dataset_id)

    context = load_context(dataset_id)

    result = get_overview(
        context.df,
        context.target
    )

    result["dataset"] = {
        "dataset_id": context.dataset_id,
        "source": context.source,
        "target": context.target,
        "positive_class": context.positive_class,
        "negative_class": context.negative_class,
        "class_selection": context.class_selection
    }

    return result


@router.get("/preview")
def preview(
    limit: int = Query(10, ge=1, le=100),
    dataset_id: str | None = None
):
    logger.info("GET /data/preview")

    context = load_context(dataset_id)

    return {
        "rows": json.loads(
            context.df.head(limit).to_json(
                orient="records"
            )
        )
    }


@router.get("/correlations")
def correlations(dataset_id: str | None = None):
    logger.info("GET /data/correlations")

    context = load_context(dataset_id)
    return get_correlations(context.df)


@router.get("/feature/{column}")
def feature_analysis(
    column: str,
    dataset_id: str | None = None
):
    logger.info("GET /data/feature/%s", column)

    context = load_context(dataset_id)

    try:
        return get_feature_analysis(
            context.df,
            column,
            context.target,
            context.positive_class
        )

    except ValueError as exc:
        logger.warning("Erreur EDA : %s", exc)
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc
