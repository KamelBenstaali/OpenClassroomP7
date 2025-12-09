# Frontend Streamlit
Interface utilisateur Streamlit pour envoyer des textes à l'API FastAPI et signaler les prédictions incorrectes. Inclut un habillage visuel (héros, badges) et un petit historique local des requêtes.

## Prérequis
- Python 3.10+ et installation `pip install -r requirements.txt` (Streamlit, Requests).
- Backend lancé et accessible (par défaut sur `http://localhost:8000`).
- Assets stockés dans `app/FrontEnd/asserts` (`AirParadis_logo.png`, `AirParadis_landing.png`).

## Configuration
- `PREDICT_API_URL` : URL de `POST /predict` (par défaut `http://localhost:8000/predict`).
- `FEEDBACK_API_URL` : URL de `POST /feedback` (déduite de `PREDICT_API_URL` si non définie).
- `HISTORY_FILE_PATH` : chemin du fichier d'historique local (défaut `app/FrontEnd/conversation_history.txt`).
- `APP_TEST_MODE=1` : désactive le rendu complet pour les exécutions de tests automatisés.

## Démarrage
Depuis `app/FrontEnd` : `streamlit run streamlit_app.py --server.port 8501`  
L'interface se charge sur `http://localhost:8501`. Assurez-vous que l'API tourne avant d'envoyer une requête.

## Fonctionnalités clés
- Zone de saisie avec exemples pré-remplis.
- Appel API avec spinner, affichage du label + score, badges colorés.
- Formulaire de feedback envoyant un signalement à l'API et persistance de l'historique (bouton pour vider).
- Fond illustré et thème personnalisé injecté en CSS pour un rendu plus engageant.
