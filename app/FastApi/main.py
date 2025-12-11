# pour run l'appi faire ces 2 commandes en terminal:
# (1) Mettre la chaîne dans un fichier .env (non commité) :
#     APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=...;LiveEndpoint=...;ApplicationId=..."
# (2) Lancer : uvicorn main:app --reload --port 8000
import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer

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

MODEL_DIR = Path(__file__).resolve().parent  # model colocated with this file
ONNX_MODEL_PATH = MODEL_DIR / "model-int8-static.onnx"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
DEFAULT_TOKENIZER_ID = "distilbert-base-uncased"
MAX_SEQ_LEN = 128  # sequence length used for ONNX export/quantization
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}


class OnnxTextClassifier:
    """Minimal wrapper to run ONNXRuntime inference with the existing tokenizer."""

    def __init__(self, tokenizer, session: ort.InferenceSession, id2label: Dict[int, str], max_length: int):
        self.tokenizer = tokenizer
        self.session = session
        self.id2label = id2label
        self.max_length = max_length
        self.output_names = [o.name for o in session.get_outputs()]

    def __call__(self, text: str):
        encoded = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        feed = {}
        for input_meta in self.session.get_inputs():
            name = input_meta.name
            base = name.split(":")[0]
            if base in encoded:
                feed[name] = encoded[base]
        outputs = self.session.run(self.output_names, feed)
        logits = np.asarray(outputs[0])
        probs = self._softmax(logits)
        probs_vector = probs if probs.ndim == 1 else probs[0]
        best_idx = int(np.argmax(probs_vector))
        best_score = float(probs_vector[best_idx])
        label = self.id2label.get(best_idx, f"LABEL_{best_idx}")
        return [{"label": label, "score": best_score}]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=-1, keepdims=True)


def build_classifier():
    """Return inference pipeline backed by the quantized ONNX model."""
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found at {ONNX_MODEL_PATH}")

    if TOKENIZER_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    else:
        # Fallback to hub tokenizer if local assets were not copied next to the ONNX file.
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_ID)
    id2label = {int(k): v for k, v in ID2LABEL.items()}

    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    return OnnxTextClassifier(tokenizer, session, id2label, max_length=MAX_SEQ_LEN)


# load once at startup
clf = build_classifier()

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "positive",
    "NEGATIVE": "negative",
    "POSITIVE": "positive",
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
