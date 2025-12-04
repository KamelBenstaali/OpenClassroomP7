import os
import requests
import streamlit as st

# API endpoint for the FastAPI /predict route.
API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict")
FEEDBACK_API_URL = os.getenv(
    "FEEDBACK_API_URL", API_URL.replace("/predict", "/feedback")
)

st.set_page_config(page_title="Sentiment Demo", page_icon="🤖", layout="centered")

st.title("Sentiment Classification")
st.caption(f"Backend: {API_URL}")
st.caption(f"Feedback API: {FEEDBACK_API_URL}")

example_texts = [
    "I love how friendly this app is!",
    "The experience was terrible and frustrating.",
    "It works but I'm not impressed.",
]

text = st.text_area("Votre texte", height=140, value=example_texts[0])

col1, col2 = st.columns([1, 1])
with col1:
    use_example = st.selectbox("Exemples", options=["(aucun)"] + example_texts, index=1)
with col2:
    submit = st.button("Analyser", use_container_width=True)

if use_example != "(aucun)":
    text = use_example

if "last_pred" not in st.session_state:
    st.session_state["last_pred"] = None
if "feedback_status" not in st.session_state:
    st.session_state["feedback_status"] = None


def call_api(payload: dict):
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as exc:
        return None, str(exc)


def send_feedback(payload: dict):
    try:
        resp = requests.post(FEEDBACK_API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as exc:
        return None, str(exc)


if submit:
    if not text.strip():
        st.warning("Merci de saisir un texte.")
    else:
        with st.spinner("Analyse en cours..."):
            data, err = call_api({"text": text})
        if err:
            st.error(f"Erreur d'appel API : {err}")
            st.session_state["last_pred"] = None
        else:
            st.session_state["last_pred"] = {
                "text": text,
                "label": data.get("label"),
                "score": data.get("score"),
            }
            st.session_state["feedback_status"] = None
            st.success(f"Label : {data.get('label')}  |  Score : {data.get('score'):.4f}")

last_pred = st.session_state.get("last_pred")

if last_pred:
    st.divider()
    st.info("La prédiction vous semble incorrecte ? Signalez-la pour améliorer le modèle.")

    with st.form("feedback_form"):
        st.write(
            f"Dernière prédiction : **{last_pred['label']}** "
            f"(score {last_pred['score']:.4f})"
        )
        expected_label = st.selectbox(
            "Label attendu (optionnel)",
            options=["(inconnu)", "positive", "negative"],
            index=0,
        )
        comment = st.text_area(
            "Commentaire (optionnel)",
            height=80,
            placeholder="Pourquoi pensez-vous que la prédiction est incorrecte ?",
        )
        send = st.form_submit_button("Signaler une mauvaise prédiction", use_container_width=True)

    if send:
        payload = {
            "text": last_pred["text"],
            "predicted_label": last_pred["label"],
            "score": last_pred["score"],
            "expected_label": expected_label if expected_label != "(inconnu)" else None,
            "comment": comment or None,
        }
        with st.spinner("Envoi du signalement..."):
            feedback_resp, feedback_err = send_feedback(payload)
        if feedback_err:
            st.error(f"Impossible d'envoyer le feedback : {feedback_err}")
            st.session_state["feedback_status"] = "error"
        else:
            st.success("Merci, votre signalement a été transmis.")
            st.session_state["feedback_status"] = "sent"

st.divider()
st.markdown(
    """
    **Instructions**  
    1. Lancez d'abord le backend FastAPI : `uvicorn app.FastApi.main:app --reload --port 8000`  
    2. Démarrez Streamlit : `streamlit run app/FrontEnd/streamlit_app.py --server.port 8501`  
    3. Ouvrez: http://localhost:8501  
    (Changez `PREDICT_API_URL` ou `FEEDBACK_API_URL` dans l'environnement si l'API écoute ailleurs.)
    """
)
