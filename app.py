import streamlit as st
import joblib

model = joblib.load('fake_review_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

st.title("🛡️ Fake Review Detector")
st.subheader("Enter a product review below:")

review = st.text_area("Review", height=150)

if st.button("Check Review"):
    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        vector = tfidf.transform([review])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0]
        
        if prediction == 1:
            st.error(f"❌ FAKE Review — Confidence: {confidence[1]*100:.1f}%")
        else:
            st.success(f"✅ GENUINE Review — Confidence: {confidence[0]*100:.1f}%")