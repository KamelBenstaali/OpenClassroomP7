# pour run l'appi faire ces 2 commandes en terminal:
# export APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=9263ed5f-8e15-4adc-870c-88b9b6991aa2;IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus.livediagnostics.monitor.azure.com/;ApplicationId=f4e116ca-26f4-4252-b673-57e2fcbb34c1"
# run with: uvicorn main:app --reload --port 8000
from pathlib import Path
import re
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import logging
# Import the `configure_azure_monitor()` function from the
# `azure.monitor.opentelemetry` package.
from azure.monitor.opentelemetry import configure_azure_monitor

# Configure OpenTelemetry to use Azure Monitor with the 
# APPLICATIONINSIGHTS_CONNECTION_STRING environment variable.
configure_azure_monitor(
    logger_name="DefaultWorkspace-cb3c0f01-20ec-4646-90f4-acaa0bbd95ca-EUS",  # Set the namespace for the logger in which you would like to collect telemetry for if you are collecting logging telemetry. This is imperative so you do not collect logging telemetry from the SDK itself.
)
logger = logging.getLogger("DefaultWorkspace-cb3c0f01-20ec-4646-90f4-acaa0bbd95ca-EUS")  # Logging telemetry will be collected from logging calls made with this logger and all of it's children loggers.

MODEL_ROOT = Path("/home/kamel/Openclassroom_projets/P7/Mes_notebooks/Model_4_DISTILBERT/distilbert_model_package")


# load once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT / "tokenizer")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ROOT / "hf_model")
clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)  # set to 0 if you have GPU

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
    friendly_label = LABEL_MAP.get(result["label"], result["label"])
    return {
        "label": friendly_label,
        "score": result["score"],
    }


@app.post("/feedback")
def send_feedback(feedback: Feedback):
    # Logging telemetry will be sent to Azure Application Insights via configure_azure_monitor.
    logger.warning(
        "prediction_feedback_reported",
        extra={
            "custom_dimensions": {
                "text": preprocess(feedback.text),
                "predicted_label": feedback.predicted_label,
                "score": feedback.score,
                "expected_label": feedback.expected_label,
                "comment": feedback.comment,
            }
        },
    )
    return {"status": "ok"}
