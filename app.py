import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

model = joblib.load('fake_review_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

st.title("🛡️ Fake Review Detector")
st.subheader("Enter a product review:")

# Initialize session state for the text area
if 'review_text' not in st.session_state:
    st.session_state.review_text = ""

# Create two columns for buttons
col1, col2 = st.columns([0.8, 0.2])

review = st.text_area("Review", height=150, value=st.session_state.review_text, key='review_input')

# Button layout
col1, col2 = st.columns(2)

with col1:
    if st.button("Check Review", use_container_width=True):
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

with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.review_text = ""
        st.rerun()