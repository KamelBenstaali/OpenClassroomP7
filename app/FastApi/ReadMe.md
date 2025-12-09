# Backend FastAPI
API de classification de sentiments exposée avec FastAPI. Elle charge un modèle DistilBERT déjà fine-tuné pour classer un texte en `positive` ou `negative` et envoie les retours utilisateurs vers Application Insights.

## Prérequis
- Python 3.10+ et les dépendances listées dans `requirements.txt` (FastAPI, Transformers, python-dotenv, azure-monitor-opentelemetry, etc.).
- Modèle sauvegardé dans `Mes_notebooks/Model_4_DISTILBERT/distilbert_model_package` (tokenizer + `hf_model`) accessible au chemin absolu défini dans `main.py`.
- Un fichier `.env` (non commité) contenant au minimum `APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=...;LiveEndpoint=...;ApplicationId=..."`. En local sans Azure, définir `APP_TEST_MODE=1` pour utiliser une chaîne factice.

## Lancement local
1) (Optionnel) `python -m venv .venv && source .venv/bin/activate`
2) `pip install -r requirements.txt`
3) Depuis `app/FastApi`, démarrer l'API : `uvicorn main:app --reload --port 8000`

## Endpoints principaux
- `POST /predict`  
  Corps : `{"text": "<votre phrase>"}`  
  Réponse : `{"label": "positive|negative", "score": <float>}`
- `POST /feedback`  
  Corps : `{"text": "...", "predicted_label": "...", "score": <float>, "expected_label": "positive|negative", "comment": "..."}`  
  Envoie un événement `prediction_feedback_reported` aux logs applicatifs (collectés par Application Insights si configuré).

## Notes techniques
- CORS autorise les origines du front Streamlit (`http://localhost:8501` et `http://127.0.0.1:8501`).
- Les chaînes sont nettoyées (trim + réduction des espaces) avant prédiction et avant envoi des métadonnées de feedback.
- Des commandes pour tester la consommation CPU/mémoire se trouvent dans `app/FastApi/test_performance_ressources/commandes pour test ressources.txt`.
