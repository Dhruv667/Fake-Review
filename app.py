import streamlit as st
import joblib

model = joblib.load('fake_review_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

st.set_page_config(page_title="Review Authenticity Checker", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <div style='text-align: center; padding: 20px 0; margin-bottom: 30px;'>
        <h1 style='font-size: 2.5em; color: #3b82f6; margin: 0;'>🛡️ Fake Review Detector</h1>
        <p style='color: #94a3b8; font-size: 1em; margin: 10px 0 0 0;'>Detect fake reviews with machine learning</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("### Enter Review Text")
review_input = st.text_area("Review", value=st.session_state.review_text, placeholder="Paste the review here...", height=100, label_visibility="collapsed")
st.session_state.review_text = review_input

col1, col2 = st.columns(2)
with col1:
    analyze_button = st.button("🔍 Check Review", use_container_width=True, key="analyze")
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True, key="clear")

if clear_button:
    st.session_state.review_text = ""
    st.rerun()

st.divider()

if analyze_button and review_input.strip():
    vector = tfidf.transform([review_input])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]

    if prediction == 1:
        confidence = proba[1] * 100
        result_label = "FAKE"
    else:
        confidence = proba[0] * 100
        result_label = "GENUINE"

    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        if result_label == "GENUINE":
            st.success("✅ GENUINE")
        else:
            st.error("❌ FAKE")
    with col2:
        st.metric("Confidence", f"{confidence:.0f}%")
    with col3:
        level = "High" if result_label == "GENUINE" else "Low"
        st.metric("Authenticity", level)

    st.progress(confidence / 100)

    st.divider()
    st.markdown("### 📊 Analysis Details")
    col_left, col_right = st.columns(2)
    with col_left:
        st.write(f"**Prediction:** {result_label}")
        st.write(f"**Model confidence:** {confidence:.1f}%")
        st.write(f"**FAKE probability:** {proba[1]*100:.1f}%")
    with col_right:
        st.write(f"**GENUINE probability:** {proba[0]*100:.1f}%")
        st.write(f"**Word count:** {len(review_input.split())}")

    st.divider()
    if result_label == "FAKE":
        st.error("**🚨 This review appears FAKE**\n\nBased on the trained model's analysis of review text patterns.")
    else:
        st.success("**✅ This review appears GENUINE**\n\nBased on the trained model's analysis of review text patterns.")
elif analyze_button and not review_input.strip():
    st.error("⚠️ Please enter a review first!")
