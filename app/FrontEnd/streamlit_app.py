import os
import requests
import streamlit as st

# API endpoint for the FastAPI /predict route.
API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Sentiment Demo", page_icon="🤖", layout="centered")

st.title("Sentiment Classification")
st.caption(f"Backend: {API_URL}")

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


def call_api(payload: dict):
    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
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
        else:
            st.success(f"Label : {data.get('label')}  |  Score : {data.get('score'):.4f}")

st.divider()
st.markdown(
    """
    **Instructions**  
    1. Lancez d'abord le backend FastAPI : `uvicorn app.FastApi.main:app --reload --port 8000`  
    2. Démarrez Streamlit : `streamlit run app/FrontEnd/streamlit_app.py --server.port 8501`  
    3. Ouvrez: http://localhost:8501  
    (Changez `PREDICT_API_URL` dans l'environnement si l'API écoute ailleurs.)
    """
)
