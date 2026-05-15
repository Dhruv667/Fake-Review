import streamlit as st
import joblib
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

    return nb_model, lr_model, vectorizer, model_info

try:
    nb_model, lr_model, vectorizer, model_info = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("🔍 Fake Review Detection System")
st.subheader("Amazon Fake Review Detection using Machine Learning")

st.markdown("---")

sample_reviews = {
    "Fake Review": "BEST PRODUCT EVER!!! AMAZING QUALITY!!! MUST BUY RIGHT NOW!!!",
    "Genuine Review": "The product quality is decent for the price. Delivery was fast and packaging was good."
}

if "review_input" not in st.session_state:
    st.session_state.review_input = ""

col1, col2 = st.columns(2)

with col1:
    if st.button("📌 Fake Sample", use_container_width=True):
        st.session_state.review_input = sample_reviews["Fake Review"]

with col2:
    if st.button("📌 Genuine Sample", use_container_width=True):
        st.session_state.review_input = sample_reviews["Genuine Review"]

review_input = st.text_area(
    "📝 Enter Review",
    value=st.session_state.review_input,
    height=200,
    placeholder="Paste your review here..."
)

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

        st.markdown("## 📊 Results")

        tabs = st.tabs([
            "🤖 Naive Bayes",
            "📈 Logistic Regression",
            "🎯 Final Result"
        ])

        with tabs[0]:

            if nb_pred == 1:
                st.success(f"GENUINE REVIEW\n\nConfidence: {nb_conf:.2f}%")
            else:
                st.error(f"FAKE REVIEW\n\nConfidence: {nb_conf:.2f}%")

        with tabs[1]:

            if lr_pred == 1:
                st.success(f"GENUINE REVIEW\n\nConfidence: {lr_conf:.2f}%")
            else:
                st.error(f"FAKE REVIEW\n\nConfidence: {lr_conf:.2f}%")

        with tabs[2]:

            predictions = [nb_pred, lr_pred]

            final_pred = 1 if sum(predictions) >= 1 else 0

            if final_pred == 1:
                st.success("✅ FINAL RESULT: GENUINE REVIEW")
            else:
                st.error("❌ FINAL RESULT: FAKE REVIEW")

st.markdown("---")

st.markdown(
    "<div style='text-align:center;'>Fake Review Detection System using Machine Learning</div>",
    unsafe_allow_html=True
)