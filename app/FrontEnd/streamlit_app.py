import json
import os
from base64 import b64encode
from html import escape
from pathlib import Path

import requests
import streamlit as st

# Flag pour désactiver le rendu complet pendant les tests.
APP_TEST_MODE = os.getenv("APP_TEST_MODE") == "1"

# API endpoints
API_URL = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict")
FEEDBACK_API_URL = os.getenv("FEEDBACK_API_URL", API_URL.replace("/predict", "/feedback"))

ASSETS_DIR = Path(__file__).resolve().parent / "asserts"
ICON_PATH = str(ASSETS_DIR / "AirParadis_logo.png")
HERO_BG = ASSETS_DIR / "AirParadis_landing.png"
HISTORY_FILE = Path(
    os.getenv(
        "HISTORY_FILE_PATH",
        str(Path(__file__).resolve().parent / "conversation_history.txt"),
    )
)


def to_data_uri(path: Path) -> str:
    try:
        data = path.read_bytes()
        return "data:image/png;base64," + b64encode(data).decode("utf-8")
    except Exception:
        return ""


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


def persist_history(entry: dict):
    """Append prediction entry to a local history file (one JSON per line)."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            json.dump(
                {
                    "text": entry.get("text"),
                    "label": entry.get("label"),
                    "score": entry.get("score"),
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    except Exception:
        # Don't block UI on persistence errors
        pass


def render_app():
    HERO_BG_DATAURI = to_data_uri(HERO_BG)

    st.set_page_config(page_title="Sentiment Demo", page_icon=ICON_PATH, layout="centered")

    # ---- Custom styles ----
    st.markdown(
        """
        <style>
        /* Layout */
        .main {background: radial-gradient(circle at 20% 20%, #0f2d3f 0, #0a1620 45%, #050b13 100%); color: #e9eef5;}
        section[data-testid="stSidebar"] {background: #0b1826;}
        /* Cards */
        .card {background: rgba(255,255,255,0.04); border: 1px solid #1dd1a1; border-radius: 16px; padding: 18px 18px 14px 18px; margin-bottom: 14px;}
        /* Inputs */
        textarea, select, input, .stButton>button {border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.12) !important;}
        textarea, select, input {
            background: #f8fbff !important;
            color: #0a0a0a !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
        }
        textarea::placeholder, input::placeholder {color: rgba(0,0,0,0.5) !important;}
        .stButton>button {background: linear-gradient(120deg, #00a6ff, #1dd1a1); color: #04101a; font-weight: 700; border: none;}
        .stButton>button:hover {opacity: 0.92;}
        /* Titles */
        h1, h2, h3, h4, h5, h6 {color: #e9eef5;}
        h1 {font-weight: 800; letter-spacing: -0.5px;}
        /* Alert/info blocks */
        .info-banner {background: rgba(0,166,255,0.12); border: 1px solid rgba(0,166,255,0.3); border-radius: 12px; padding: 10px 12px; color: #000; text-align: center; margin-bottom: 14px;}
        .badge {padding: 6px 10px; border-radius: 10px; font-weight: 700; font-size: 13px;}
        .badge-positive {background: rgba(29,209,161,0.18); color: #7bf3cc; border: 1px solid rgba(29,209,161,0.4);}
        .badge-negative {background: rgba(255,99,132,0.18); color: #ffb3c4; border: 1px solid rgba(255,99,132,0.4);}
        .badge-neutral {background: rgba(255,255,255,0.12); color: #f1f5f9; border: 1px solid rgba(255,255,255,0.2);}
        /* Hero */
        .hero {
            background: rgba(255,255,255,0.04);
            border-radius: 18px;
            padding: 18px 18px 12px 18px;
            margin-bottom: 14px;
            margin-top:auto;
            border: 1px solid #1dd1a1;
            text-align: center;
        }
        .hero-img-box {margin: 0 0 12px 0; width: 100%; display: flex; justify-content: center;}
        .hero-img {
            width: 100%;
            max-width: 960px;
            height: auto;
            max-height: 450px;
            object-fit: contain; /* keep entire image */
            background: transparent;
            border-radius: 16px;
            border: none;
            display: block;
            margin: 0 auto;
            padding: 0;
        }
        .hero h3 {margin: 0; padding: 0;}
        .hero p {margin: 6px 0 0 0; color: #000; font-size: 15px; line-height: 1.5;}
        /* History */
        .history-card {padding: 16px;}
        .history-item {padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 10px; background: rgba(255,255,255,0.02);}
        .history-header {display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; color:#e9eef5;}
        .history-text {
            background: rgba(255,255,255,0.04);
            padding: 10px;
            border-radius: 10px;
            color: #000;
            white-space: pre-wrap;       /* keep user spacing but wrap lines */
            word-break: break-word;      /* break long tokens */
            overflow-wrap: anywhere;     /* ensure overflow never spills */
            max-width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Hero / header ----
    st.markdown(
        f"""
        <div class="hero-img-box" style="width:100%;">
            <img src="{HERO_BG_DATAURI}" class="hero-img" alt="Air Paradis landing" />
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
            <h3>Prédiction de sentiment</h3>
            <p>DistilBERT finetuné pour classer vos textes en positif / négatif. Signalez les erreurs pour améliorer le modèle.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "last_pred" not in st.session_state:
        st.session_state["last_pred"] = None
    if "feedback_status" not in st.session_state:
        st.session_state["feedback_status"] = None
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "show_history" not in st.session_state:
        st.session_state["show_history"] = False

    example_texts = [
        "I love how friendly this app is!",
        "The experience was terrible and frustrating.",
        "It works but I'm not impressed.",
    ]
    text = example_texts[0]
    show_history = st.session_state["show_history"]
    submit = False
    use_example = "(aucun)"

    # ---- Input section ----
    st.divider()
    st.markdown("#### Votre texte")
    with st.container():
        col_text, col_action = st.columns([3, 1])
        with col_text:
            text = st.text_area(" ", height=150, label_visibility="collapsed", placeholder="Saisissez un tweet à analyser ici...")
        with col_action:
            st.markdown("###### Exemples")
            use_example = st.selectbox(" ", options=["(aucun)"] + example_texts, index=0, label_visibility="collapsed")
            submit = st.button("Analyser", use_container_width=True)

    if use_example != "(aucun)":
        text = use_example

    # ---- Prediction handling ----
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
                new_entry = {
                    "text": text,
                    "label": data.get("label"),
                    "score": data.get("score"),
                }
                st.session_state["history"].insert(
                    0,
                    new_entry,
                )
                # Keep a small buffer to avoid unbounded growth in session
                st.session_state["history"] = st.session_state["history"][:50]
                persist_history(new_entry)
                st.session_state["feedback_status"] = None

    last_pred = st.session_state.get("last_pred")

    # ---- Prediction card ----
    if last_pred:
        label = last_pred["label"] or "inconnu"
        score_val = last_pred["score"] or 0
        score = f"{score_val:.4f}"
        badge_class = "badge-positive" if label.lower() == "positive" else "badge-negative"
        st.markdown(
            f"""
            <div class="card">
                <div style="display:flex;align-items:center;justify-content:space-between;">
                    <div style="font-size:18px;font-weight:700;">Prédiction</div>
                    <div class="badge {badge_class}">{label.upper()}</div>
                </div>
                <div style="margin-top:10px;color:#000;">Score : {score}</div>
                <div style="margin-top:10px;color:#000;">Texte analysé :</div>
                <div style="margin-top:4px; background:rgba(255,255,255,0.03); padding:10px; border-radius:10px; color:#000;">{last_pred['text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Feedback section ----
    if last_pred:
        st.markdown(
            """
            <div class="info-banner">
                La prédiction vous semble incorrecte ? Signalez-la pour améliorer le modèle.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("feedback_form"):
            st.markdown(
                f"Dernière prédiction : **{label.upper()}** (score {score})",
                help="Référence sur laquelle portera votre signalement.",
            )
            expected_label = st.selectbox(
                "Label attendu",
                options=["positive", "negative"],
                index=0,
            )
            comment = st.text_area(
                "Commentaire (optionnel)",
                height=90,
                placeholder="Pourquoi pensez-vous que la prédiction est incorrecte ?",
            )
            send = st.form_submit_button("Signaler une prédiction", use_container_width=True)

        if send:
            payload = {
                "text": last_pred["text"],
                "predicted_label": last_pred["label"],
                "score": last_pred["score"],
                "expected_label": expected_label,
                "comment": comment or None
            }
            with st.spinner("Envoi du signalement..."):
                feedback_resp, feedback_err = send_feedback(payload)
            if feedback_err:
                st.error(f"Impossible d'envoyer le feedback : {feedback_err}")
                st.session_state["feedback_status"] = "error"
            else:
                st.success("Merci, votre signalement a été transmis.")
                st.session_state["feedback_status"] = "sent"

    # ---- History toggle & display (below prediction/feedback) ----
    toggle_label = "Masquer l'historique" if st.session_state["show_history"] else "Afficher l'historique"
    if st.button(toggle_label, key="history_toggle", use_container_width=True):
        st.session_state["show_history"] = not st.session_state["show_history"]
        show_history = st.session_state["show_history"]
        st.rerun()
    show_history = st.session_state["show_history"]
    if show_history:
        history_items = st.session_state["history"]
        header_cols = st.columns([5, 1])
        with header_cols[0]:
            st.markdown("### Historique des prédictions")
        with header_cols[1]:
            if st.button("Vider l'historique", key="clear_history", use_container_width=True):
                st.session_state["history"] = []
                try:
                    HISTORY_FILE.unlink(missing_ok=True)
                except Exception:
                    pass
                st.rerun()

        if not history_items:
            st.info("Aucune prédiction enregistrée pour le moment.")
        else:
            history_html = '<div class="card history-card">'
            for idx, item in enumerate(history_items, start=1):
                hist_label = (item.get("label") or "inconnu")
                hist_badge = "badge-positive" if hist_label.lower() == "positive" else "badge-negative"
                hist_text = escape(item.get("text") or "")
                history_html += (
                    '<div class="history-item">'
                    '<div class="history-header">'
                    f'<div class="history-text">{hist_text}</div>'
                    f'<div class="badge {hist_badge}">{hist_label.upper()}</div>'
                    "</div>"
                    "</div>"
                )
            history_html += "</div>"
            st.markdown(history_html, unsafe_allow_html=True)


if not APP_TEST_MODE:
    render_app()
