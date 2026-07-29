import streamlit as st
import joblib

# Load ML model
model = joblib.load("fake_review_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

st.set_page_config(
    page_title="Review Authenticity Checker",
    page_icon="🔍",
    layout="centered",
)

# Header
st.markdown("""
<div style="text-align:center;padding:15px;">
    <h1 style="color:#3b82f6;">🔍 Review Authenticity Checker</h1>
    <p style="color:gray;">
        Analyze product reviews using NLP & Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.title("About")
    st.write("""
This application uses a machine learning-based classification engine to detect whether a product review is **Fake** or **Genuine**.

### Tech Stack
- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorization
- MLP Neural Network
""")

st.subheader("Enter Review")

review = st.text_area(
    "",
    value=st.session_state.review_text,
    placeholder="Paste the review here...",
    height=150,
)

st.session_state.review_text = review

c1, c2 = st.columns(2)

with c1:
    check = st.button("🔍 Analyze", use_container_width=True)

with c2:
    clear = st.button("🗑️ Clear", use_container_width=True)

if clear:
    st.session_state.review_text = ""
    st.rerun()

if check:

    if review.strip() == "":
        st.warning("⚠️ Please enter a review.")
        st.stop()

    vector = tfidf.transform([review])

    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]

    if prediction == 1:
        label = "FAKE"
        confidence = probability[1] * 100
        fake_prob = probability[1] * 100
        genuine_prob = probability[0] * 100
    else:
        label = "GENUINE"
        confidence = probability[0] * 100
        fake_prob = probability[1] * 100
        genuine_prob = probability[0] * 100

    st.divider()

    a, b, c = st.columns(3)

    with a:
        if label == "FAKE":
            st.error("❌ FAKE")
        else:
            st.success("✅ GENUINE")

    with b:
        st.metric("Confidence", f"{confidence:.1f}%")

    with c:
        st.metric("Words", len(review.split()))

    st.progress(confidence / 100)

    st.info("🤖 Prediction generated using TF-IDF + Machine Learning.")

    st.divider()

    st.subheader("📊 Analysis Details")

    left, right = st.columns(2)

    with left:
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence:.1f}%")
        st.write(f"**Word Count:** {len(review.split())}")

    with right:
        st.write(f"**Fake Probability:** {fake_prob:.1f}%")
        st.write(f"**Genuine Probability:** {genuine_prob:.1f}%")
        st.write(f"**Characters:** {len(review)}")

    st.divider()

    if label == "FAKE":
        st.error(
            "🚨 **This review appears to be FAKE.**\n\n"
            "The trained machine learning model detected patterns commonly associated with deceptive reviews."
        )
    else:
        st.success(
            "✅ **This review appears to be GENUINE.**\n\n"
        )

st.markdown("---")
st.caption("Developed by Dhruv Bhoir | Fake Review Detection System")
