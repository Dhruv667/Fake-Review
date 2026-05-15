import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

model = joblib.load('fake_review_model_final.pkl')
tfidf = joblib.load('tfidf_final.pkl')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

st.title("🛡️ Fake Review Detector")
st.subheader("Enter a product review:")

review = st.text_area("Review", height=150)

if st.button("Check Review"):
    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector)[0]
        
        if prediction == 0:
            st.error(f"❌ DECEPTIVE Review — Confidence: {confidence[0]*100:.1f}%")
        else:
            st.success(f"✅ GENUINE Review — Confidence: {confidence[1]*100:.1f}%")