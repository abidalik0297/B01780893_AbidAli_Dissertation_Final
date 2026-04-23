import streamlit as st
import joblib
import torch
import numpy as np
import re

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer


# ============================
# PAGE CONFIG
# ============================

st.set_page_config(
    page_title="Phishing Detection System",
    page_icon="📧",
    layout="wide"
)


# ============================
# TEXT CLEANING (CRITICAL FIX)
# ============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


# ============================
# LOAD MODELS
# ============================

@st.cache_resource
def load_models():

    tfidf_model = joblib.load(r"D:\abid_uws\outputs\models\tfidf_logistic_model.pkl")

    bert_tokenizer = DistilBertTokenizerFast.from_pretrained(
        r"D:\abid_uws\outputs\models\distilbert_phishing"
    )

    bert_model = DistilBertForSequenceClassification.from_pretrained(
        r"D:\abid_uws\outputs\models\distilbert_phishing"
    )

    ocsvm = joblib.load(r"D:\abid_uws\outputs\models\oneclass_svm.pkl")

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    return tfidf_model, bert_tokenizer, bert_model, ocsvm, sbert


tfidf_model, bert_tokenizer, bert_model, ocsvm, sbert = load_models()


# ============================
# HEADER
# ============================

st.title("📧 AI-Based Phishing Detection System")
st.markdown("Multi-model detection using ML, Deep Learning, and Anomaly Detection")

st.divider()


# ============================
# INPUT
# ============================

col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area("✉️ Enter Email Text", height=220)

with col2:
    model_choice = st.selectbox(
        "Select Mode",
        ["Compare All", "TF-IDF", "DistilBERT", "Anomaly Detection"]
    )

    if st.button("Load Demo"):
        st.session_state["demo"] = "Hi team, let's meet tomorrow at 10 AM regarding the project."
        st.rerun()


if "demo" in st.session_state:
    user_input = st.session_state["demo"]


# ============================
# RESULT DISPLAY
# ============================

def show_result(title, pred, prob):

    pred = int(pred)
    prob = float(prob)

    if pred == 1:
        st.error(f"⚠️ {title}: Phishing Detected")
        st.progress(min(prob, 1.0))
        st.caption(f"Confidence: {prob:.2f}")
    else:
        st.success(f"✅ {title}: Legitimate Email")
        st.progress(min(1 - prob, 1.0))
        st.caption(f"Confidence: {1 - prob:.2f}")


# ============================
# ANALYZE
# ============================

if st.button("🔍 Analyze Email"):

    if user_input.strip() == "":
        st.warning("Please enter email text.")
    else:

        cleaned_input = clean_text(user_input)

        st.subheader("📊 Results")

        # ---------------- TF-IDF ----------------
        if model_choice in ["TF-IDF", "Compare All"]:

            pred = tfidf_model.predict([cleaned_input])[0]
            prob = tfidf_model.predict_proba([cleaned_input])[0][1]

            show_result("TF-IDF Model", pred, prob)

        # ---------------- BERT ----------------
        if model_choice in ["DistilBERT", "Compare All"]:

            inputs = bert_tokenizer(
                cleaned_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=96
            )

            outputs = bert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

            pred = torch.argmax(probs, dim=1).cpu().numpy()[0]
            prob = probs[0][1].item()

            show_result("DistilBERT Model", pred, prob)

        # ---------------- OCSVM ----------------
        if model_choice in ["Anomaly Detection", "Compare All"]:

            emb = sbert.encode([cleaned_input])
            pred_raw = ocsvm.predict(emb)[0]

            # Convert OCSVM output
            pred = 0 if pred_raw == 1 else 1
            prob = 0.5

            show_result("Anomaly Detection", pred, prob)

        st.divider()

        st.info("""
        **Model Explanation:**
        - TF-IDF → keyword-based detection  
        - DistilBERT → contextual understanding  
        - Anomaly Model → detects unusual patterns (zero-day attacks)
        """)


# ============================
# FOOTER
# ============================

st.divider()
st.caption("Final Year Project — AI-Based Phishing Detection System")