# Backend FastAPI
API de classification de sentiments exposée avec FastAPI. Elle charge un modèle DistilBERT quantifié en ONNX (`model-int8-static.onnx`) pour classer un texte en `positive` ou `negative` et envoie les retours utilisateurs vers Application Insights.

## Prérequis
- Python 3.10+ et les dépendances listées dans `requirements.txt` (FastAPI, ONNX Runtime, Transformers pour le tokenizer, python-dotenv, azure-monitor-opentelemetry, etc.).
- `model-int8-static.onnx` et le dossier `tokenizer/` doivent être placés dans le même répertoire que `main.py` (voir structure ci-dessous).
- Un fichier `.env` (non commité) contenant au minimum `APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=...;LiveEndpoint=...;ApplicationId=..."`. En local sans Azure, définir `APP_TEST_MODE=1` pour utiliser une chaîne factice.

## Lancement local
1) (Optionnel) `python -m venv .venv && source .venv/bin/activate`
2) `pip install -r requirements.txt`
3) Depuis `app/FastApi`, démarrer l'API : `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

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
- L'API n'a pas besoin des poids Hugging Face d'origine : seul `model-int8-static.onnx` et `tokenizer/` sont utilisés.

## Structure attendue du répertoire `app/FastApi`
```
app/FastApi/
├── main.py
├── model-int8-static.onnx
├── tokenizer/
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
├── requirements.txt
└── ReadMe.md
```

## Déploiement Azure App Service (Linux, plan F1)
1) CLI : `az login`
2) Ressources (exemple région westeurope) :
   ```
   az group create -n p7-rg -l westeurope
   az appservice plan create -n p7-plan -g p7-rg --sku F1 --is-linux
   az webapp create -n <appname-unique> -g p7-rg --plan p7-plan --runtime "PYTHON:3.10"
   ```
3) Variables d'environnement :
   ```
   az webapp config appsettings set -g p7-rg -n <appname> --settings \
     APPLICATIONINSIGHTS_CONNECTION_STRING="InstrumentationKey=...;IngestionEndpoint=...;LiveEndpoint=...;ApplicationId=..." \
     WEBSITES_PORT=8000 \
     APP_TEST_MODE=0
   ```
   (Mettre `APP_TEST_MODE=1` si vous n'avez pas la connexion Application Insights.)
4) Commande de démarrage :
   ```
   az webapp config set -g p7-rg -n <appname> \
     --startup-file "python -m uvicorn app.FastApi.main:app --host 0.0.0.0 --port 8000"
   ```
5) Déploiement en zip depuis la racine du repo :
   ```
   zip -r ../p7-fastapi.zip .
   az webapp deploy -g p7-rg -n <appname> --src-path ../p7-fastapi.zip --type zip
   ```
   Vérifiez que le zip contient `app/FastApi/model-int8-static.onnx` et `app/FastApi/tokenizer/`.
6) Logs : `az webapp log tail -g p7-rg -n <appname>` puis tester `https://<appname>.azurewebsites.net/predict` en POST JSON.
