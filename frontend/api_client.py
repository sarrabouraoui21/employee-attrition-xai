
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv(
    Path(__file__).resolve().parents[1] / ".env"
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")


# ==========================================
# GESTION DES REPONSES
# ==========================================

def check_response(response):
    if not response.ok:
        try:
            detail = response.json().get(
                "detail", response.text
            )
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"API {response.status_code} : {detail}"
        )

    return response.json()


# ==========================================
# GET - RECUPERATION DES DONNEES
# ==========================================

def api_get(endpoint, params=None):
    response = requests.get(
        f"{API_BASE_URL}/{endpoint.lstrip('/')}",
        params=params,
        timeout=120
    )

    return check_response(response)


# ==========================================
# POST - PREPROCESSING / TRAINING
# ==========================================

def api_post(endpoint, payload):
    response = requests.post(
        f"{API_BASE_URL}/{endpoint.lstrip('/')}",
        json=payload,
        timeout=180
    )

    return check_response(response)


# ==========================================
# UPLOAD - IMPORT CSV
# ==========================================

def api_upload(endpoint, uploaded_file, fields=None):
    response = requests.post(
        f"{API_BASE_URL}/{endpoint.lstrip('/')}",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "text/csv"
            )
        },
        data=fields or {},
        timeout=120
    )

    return check_response(response)
