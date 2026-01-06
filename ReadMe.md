# P7 – Analyse de sentiments (FastAPI + Streamlit)

Application complète de classification de sentiments sur des textes courts : un backend FastAPI servant un modèle DistilBERT quantifié en ONNX, un frontend Streamlit pour la démonstration et la collecte de retours, des tests automatisés et les notebooks d’entraînement/quantification.

## Description de la mission
- Contexte : ingénieur IA chez MIC (cabinet marketing). Air Paradis veut anticiper les bad buzz sur Twitter en suivant la perception des clients.
- Objectif : livrer un prototype fonctionnel d’analyse de sentiments sur tweets en comparant 4 pistes : régression logistique + TF‑IDF/features, BiLSTM avec embeddings pré-entraînés, encodeur de phrases USE + réseau dense, DistilBERT fine-tuné (export ONNX/TFLite).
- Démarche MLOps : suivi des expériences avec MLflow, CI/CD GitHub Actions vers Azure App Service, tests automatisés backend/frontend, monitoring et feedbacks utilisateur via Azure Application Insights.

## Arborescence rapide
- `app/FastApi/` : API FastAPI (`main.py`), modèle ONNX quantifié (`model-int8-static.onnx`), tokenizer et doc de déploiement Azure App Service.
- `app/FrontEnd/` : interface Streamlit (`streamlit_app.py`), assets (`asserts/`), historique local.
- `app/doc_Azure_App_Insights/` : requêtes Kusto pour suivre les feedbacks dans Application Insights.
- `app/test_performance_ressources/` : commandes pour mesurer l’empreinte mémoire/CPU en local.
- `Mes_notebooks/` : notebooks et artefacts d’entraînement (logreg, BiLSTM, USE, DistilBERT) + scripts de benchmark/quantification (ONNX, TFLite, LiteRT).
- `Mes_notebooks/Model_4_DISTILBERT_quant/` : quantification ONNX int8 (dynamique/statique) via `quantize_model.py` et notebook `testing_quant_model.ipynb` pour tester le modèle exporté.
- `sentiment140/` : données d’entraînement (brutes + versions lemmatisées/stemmées).
- `tests/` : tests Pytest pour l’API et l’app Streamlit.
- `Workflows_conception/` : notes sur les pistes de modèles.

## Prérequis
- Python 3.10+.
- Modèle et tokenizer présents dans `app/FastApi/` (fourni dans le repo).
- Variables d’environnement :
  - `APPLICATIONINSIGHTS_CONNECTION_STRING` (chaîne Azure App Insights). En local, mettre `APP_TEST_MODE=1` pour utiliser une valeur factice.
  - Frontend : `PREDICT_API_URL` (par défaut `http://localhost:8000/predict`), `FEEDBACK_API_URL` (dérivé), `HISTORY_FILE_PATH` (défaut `app/FrontEnd/conversation_history.txt`).

## Mise en route locale
```bash
python -m venv .venv
source .venv/bin/activate
# Dépendances backend minimales
pip install -r app/FastApi/requirements.txt
# Dépendances frontend (si besoin) ; le gros requirements.txt racine sert surtout aux notebooks
pip install streamlit requests
```

### Lancer le backend FastAPI
```bash
cd app/FastApi
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Endpoints principaux :
- `POST /predict` body `{"text": "..."}` → `{"label": "positive|negative", "score": <float>}`
- `POST /feedback` body `{"text": "...", "predicted_label": "...", "score": 0.9, "expected_label": "positive|negative", "comment": "..."}` (journalisé dans Application Insights via `prediction_feedback_reported`).

### Lancer le frontend Streamlit
```bash
cd app/FrontEnd
streamlit run streamlit_app.py --server.port 8501
```
L’UI envoie les textes à l’API, affiche le label + score, et permet de signaler une mauvaise prédiction (formulaire envoyé sur `/feedback`). Un historique local est conservé et peut être vidé depuis l’interface.

## Tests automatisés
Lancer depuis la racine :
```bash
APP_TEST_MODE=1 APPLICATIONINSIGHTS_CONNECTION_STRING=test pytest
```
- `tests/test_main.py` : prétraitement, chargement du classifieur ONNX, mapping des labels et payload de feedback.
- `tests/test_streamlit_app.py` : appels API simulés, persistance d’historique, flux utilisateur Streamlit (analyse, feedback, vidage de l’historique).

## Données et entraînement
- Corpus principal : `sentiment140/training.1600000.processed.noemoticon.csv` et versions prétraitées (`processed_data/lemmatized_tweets.csv`, `stemmed_tweets.csv`).
- Notebooks/artefacts :
  - `Model_1_simple` : logreg + vectorisations (packages sauvegardés par scaler).
  - `Model_2_advanced` : BiLSTM avec embeddings Word2Vec/GloVe (+ bench).
  - `Model_3_USE` : Universal Sentence Encoder (+ bench).
  - `Model_4_DISTILBERT` : fine-tuning DistilBERT (MLflow dans `mlruns`), export ONNX/TFLite/LiteRT via scripts `bench.py` ; `model-int8-static.onnx` utilisé par l’API vient de cette piste.
  - `Model_4_DISTILBERT_quant` : script `quantize_model.py` pour exporter en ONNX et quantifier en int8 (dynamique par défaut, statique possible avec calibration) ; notebook `testing_quant_model.ipynb` pour valider `model-int8-dynamic.onnx` (avec `model-int8.onnx.data`).
  - Variantes de quantification : `Model_4_DISTILBERT_onnx_int8`, `_tflite`, `_tflite_full_int8`, `_LiteRT` avec scripts de mesure RAM/poids.
- `Workflows_conception` contient les choix de modèles testés.

## Observabilité et performance
- Requêtes Kusto pour Application Insights : `app/doc_Azure_App_Insights/Commandes_Alertes.md` (détection des prédictions signalées comme fausses, alertes sur seuil).
- Astuces de mesure ressources (Gunicorn/uvicorn, ps) : `app/test_performance_ressources/commandes pour test ressources.txt`.

## Déploiement Azure (aperçu)
Un guide détaillé est dans `app/FastApi/ReadMe.md` : création du groupe de ressources et App Service Linux (plan F1), variables d’environnement (`APPLICATIONINSIGHTS_CONNECTION_STRING`, `WEBSITES_PORT`, `APP_TEST_MODE`), commande de démarrage `python -m uvicorn app.FastApi.main:app --host 0.0.0.0 --port 8000`, déploiement en zip en veillant à inclure le modèle ONNX et le tokenizer.
