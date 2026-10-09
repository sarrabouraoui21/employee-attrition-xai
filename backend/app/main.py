
import logging
import os

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv(
    Path(__file__).resolve().parents[2] / ".env"
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "DEBUG").upper(),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
    force=True
)

logger = logging.getLogger(__name__)

from app.api.data import router as data_router
from app.api.preprocessing import (
    router as preprocessing_router
)
from app.api.train import router as training_router
from app.api.evaluate import router as evaluation_router


app = FastAPI(
    title="Employee Attrition Analytics API",
    version="1.0.0",
    description="API de classification supervisée"
)

app.include_router(data_router)
app.include_router(preprocessing_router)
app.include_router(training_router)
app.include_router(evaluation_router)


@app.get("/health", tags=["Système"])
def health():
    logger.info("Backend opérationnel")
    return {"status": "ok"}
