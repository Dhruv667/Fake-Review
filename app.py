import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_models():

    nb_model = joblib.load("models/naive_bayes_model.pkl")
    lr_model = joblib.load("models/logistic_regression_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    model_info = joblib.load("models/model_info.pkl")

    try:
        nn_model = keras.models.load_model("models/neural_network_model.h5")
    except:
        nn_model = None

    return nb_model, lr_model, nn_model, vectorizer, model_info

try:
    nb_model, lr_model, nn_model, vectorizer, model_info = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.markdown("<h1>🔍 Fake Review Detection System</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Amazon Fake Review Detection using Machine Learning</div>", unsafe_allow_html=True)

st.markdown("---")

review_input = st.text_area(
    "📝 Enter Review",
    height=200,
    placeholder="Paste your review here..."
)

sample_reviews = {
    "Fake Review": "BEST PRODUCT EVER!!! AMAZING QUALITY!!! MUST BUY RIGHT NOW!!!",
    "Genuine Review": "The product quality is decent for the price. Delivery was fast and packaging was good."
}

col1, col2 = st.columns(2)

with col1:
    if st.button("📌 Fake Sample", use_container_width=True):
        review_input = sample_reviews["Fake Review"]

with col2:
    if st.button("📌 Genuine Sample", use_container_width=True):
        review_input = sample_reviews["Genuine Review"]

st.markdown("---")

if st.button("🔍 Analyze Review", use_container_width=True):

    if review_input.strip() == "":
        st.warning("Please enter a review")

    else:

        X_input = vectorizer.transform([review_input])

        nb_pred = nb_model.predict(X_input)[0]
        nb_prob = nb_model.predict_proba(X_input)[0]
        nb_conf = max(nb_prob) * 100

        lr_pred = lr_model.predict(X_input)[0]
        lr_prob = lr_model.predict_proba(X_input)[0]
        lr_conf = max(lr_prob) * 100

        if nn_model:
            X_dense = X_input.toarray()
            nn_prob = nn_model.predict(X_dense, verbose=0)[0][0]
            nn_pred = 1 if nn_prob > 0.5 else 0
            nn_conf = max(nn_prob, 1 - nn_prob) * 100

        st.markdown("## 📊 Results")

        tab_titles = ["🤖 Naive Bayes", "📈 Logistic Regression"]

        if nn_model:
            tab_titles.append("🧠 Neural Network")

        tab_titles.append("🎯 Final Result")

        tabs = st.tabs(tab_titles)

        with tabs[0]:

            if nb_pred == 1:
                st.markdown(
                    f"<div class='genuine-badge'>GENUINE REVIEW<br>Confidence: {nb_conf:.2f}%</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='fake-badge'>FAKE REVIEW<br>Confidence: {nb_conf:.2f}%</div>",
                    unsafe_allow_html=True
                )

        with tabs[1]:

            if lr_pred == 1:
                st.markdown(
                    f"<div class='genuine-badge'>GENUINE REVIEW<br>Confidence: {lr_conf:.2f}%</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='fake-badge'>FAKE REVIEW<br>Confidence: {lr_conf:.2f}%</div>",
                    unsafe_allow_html=True
                )

        tab_index = 2

        if nn_model:

            with tabs[2]:

                if nn_pred == 1:
                    st.markdown(
                        f"<div class='genuine-badge'>GENUINE REVIEW<br>Confidence: {nn_conf:.2f}%</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='fake-badge'>FAKE REVIEW<br>Confidence: {nn_conf:.2f}%</div>",
                        unsafe_allow_html=True
                    )

            tab_index = 3

        with tabs[tab_index]:

            predictions = [nb_pred, lr_pred]

            if nn_model:
                predictions.append(nn_pred)

            final_pred = 1 if sum(predictions) >= (len(predictions) / 2) else 0

            if final_pred == 1:
                st.success("✅ FINAL RESULT: GENUINE REVIEW")
            else:
                st.error("❌ FINAL RESULT: FAKE REVIEW")

st.markdown("---")

st.markdown("""
<div style='text-align:center; color:white; padding:20px;'>
Fake Review Detection System using Machine Learning
</div>
""", unsafe_allow_html=True)