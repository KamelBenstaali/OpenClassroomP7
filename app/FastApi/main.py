# pour run l'appi faire ces 2 commandes en terminal:
# (1) Mettre la chaîne dans un fichier .env (non commité) :
#     APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=...;LiveEndpoint=...;ApplicationId=..."
# (2) Lancer : uvicorn main:app --reload --port 8000
from pathlib import Path
import os
import re
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import logging
from dotenv import load_dotenv
# Import the `configure_azure_monitor()` function from the
# `azure.monitor.opentelemetry` package.
from azure.monitor.opentelemetry import configure_azure_monitor

# Charge automatiquement .env pour éviter d'exporter la variable à chaque lancement.
load_dotenv()
APPINSIGHTS_CONN_STR = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
TEST_MODE = os.getenv("APP_TEST_MODE") == "1"

if not APPINSIGHTS_CONN_STR:
    if TEST_MODE:
        # Valeur factice pour permettre le lancement local sans Azure.
        APPINSIGHTS_CONN_STR = (
            "InstrumentationKey=fake;IngestionEndpoint=http://localhost;"
            "LiveEndpoint=http://localhost;ApplicationId=fake"
        )
    else:
        raise RuntimeError(
            "APPLICATIONINSIGHTS_CONNECTION_STRING manquant. "
            "Ajoutez-le dans un fichier .env (non commité) ou dans l'environnement."
        )

if not TEST_MODE:
    # Configure OpenTelemetry to use Azure Monitor with the provided connection string.
    configure_azure_monitor(
        connection_string=APPINSIGHTS_CONN_STR,
        logger_name="DefaultWorkspace-cb3c0f01-20ec-4646-90f4-acaa0bbd95ca-EUS",  # namespace pour collecter le logging applicatif, pas celui du SDK lui-même.
    )
logger = logging.getLogger("DefaultWorkspace-cb3c0f01-20ec-4646-90f4-acaa0bbd95ca-EUS")  # Logging telemetry will be collected from logging calls made with this logger and all of it's children loggers.

MODEL_ROOT = Path("/home/kamel/Openclassroom_projets/P7/Mes_notebooks/Model_4_DISTILBERT/distilbert_model_package")


def build_classifier():
    """Return inference pipeline backed by the finetuned model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / "tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ROOT / "hf_model")
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)  # set to 0 if you have GPU


# load once at startup
clf = build_classifier()

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "positive",
}

def preprocess(text: str) -> str:
    text = text.strip()
    return re.sub(r"\s+", " ", text)

app = FastAPI()

origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    text: str


class Feedback(BaseModel):
    text: str
    predicted_label: str
    score: float
    expected_label: Optional[str] = None
    comment: Optional[str] = None


@app.post("/predict")
def predict(item: Item):
    cleaned = preprocess(item.text)
    result = clf(cleaned)[0]  # e.g. {'label': 'LABEL_1', 'score': 0.98}
    raw_label = result["label"]
    # Normalize label to friendly lower-case even if the model returns custom strings.
    friendly_label = LABEL_MAP.get(raw_label, LABEL_MAP.get(str(raw_label).upper(), raw_label)).lower()
    return {
        "label": friendly_label,
        "score": result["score"],
    }


@app.post("/feedback")
def send_feedback(feedback: Feedback):
    # Logging telemetry will be sent to Azure Application Insights via configure_azure_monitor.
    is_correct = (
        feedback.expected_label == feedback.predicted_label if feedback.expected_label else None
    )
    # Flatter custom dimensions so they arrivent dans Application Insights sans être sérialisés en chaîne.
    custom_dims = {
        "text": preprocess(feedback.text),
        "predicted_label": feedback.predicted_label,
        "score": feedback.score,
        "expected_label": feedback.expected_label,
        "comment": feedback.comment,
        "is_correct": is_correct,
    }
    # Supprime les clés None pour éviter des chaînes "None" ou des sérialisations inattendues.
    custom_dims = {k: v for k, v in custom_dims.items() if v is not None}
    logger.warning(
        "prediction_feedback_reported",
        extra=custom_dims,
    )
    return {"status": "ok"}
