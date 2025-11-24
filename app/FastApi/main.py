# run with: uvicorn app.FastApi.main:app --reload --port 8000
from pathlib import Path
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_ROOT = Path("/home/kamel/Openclassroom_projets/P7/Mes_notebooks/Model_5_DISTILBERT/mlruns/566727379184960707/fbeb3780189a45f9b2e5639783242b98/artifacts/model_package")

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

@app.post("/predict")
def predict(item: Item):
    cleaned = preprocess(item.text)
    result = clf(cleaned)[0]  # e.g. {'label': 'LABEL_1', 'score': 0.98}
    friendly_label = LABEL_MAP.get(result["label"], result["label"])
    return {
        "label": friendly_label,
        "score": result["score"],
    }



import requests

try:
    resp = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text": "A sample sentence"},
        timeout=10,
    )
    print(resp.status_code, resp.text)
except Exception as e:
    print("Request failed:", e)
