import streamlit as st
import joblib
import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from scipy.sparse import hstack

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("fake_review_detector_v2.pkl")  
    vectorizer = joblib.load("tfidf_vectorizer_v2.pkl")
    with open('feature_config.pkl', 'rb') as f:
        config = pickle.load(f)
    return model, vectorizer, config

model, vectorizer, config = load_models()
stop_words = config['stop_words']
sia = SentimentIntensityAnalyzer()

# ============================================================
# Feature Extraction
# ============================================================
def extract_features(text):
    """Extract linguistic features that indicate fake reviews"""
    
    text_lower = str(text).lower()
    text_clean = re.sub(r'[^a-z\s!?.]', '', text_lower)
    
    features = {}
    
    # 1. Excessive punctuation
    features['exclamation_ratio'] = text_clean.count('!') / max(len(text_clean), 1)
    features['question_ratio'] = text_clean.count('?') / max(len(text_clean), 1)
    features['punctuation_ratio'] = (text_clean.count('!') + text_clean.count('?')) / max(len(text_clean), 1)
    
    # 2. Word repetition
    words = text_clean.split()
    if len(words) > 0:
        word_freq = {}
        for w in words:
            if len(w) > 3:
                word_freq[w] = word_freq.get(w, 0) + 1
        features['word_repetition'] = max(word_freq.values()) / len(words) if word_freq else 0
    else:
        features['word_repetition'] = 0
    
    # 3. Sentiment intensity
    scores = sia.polarity_scores(text)
    features['sentiment_compound'] = scores['compound']
    features['sentiment_abs'] = abs(scores['compound'])
    
    # 4. Caps ratio
    caps_count = sum(1 for c in text if c.isupper())
    features['caps_ratio'] = caps_count / max(len(text), 1)
    
    # 5. Review length
    features['review_length'] = len(text)
    
    # 6. Superlatives
    superlatives = ['best', 'amazing', 'perfect', 'excellent', 'incredible', 
                    'outstanding', 'fantastic', 'wonderful', 'awesome', 'superb']
    features['superlative_count'] = sum(text_lower.count(s) for s in superlatives)
    
    # 7. First person pronouns
    features['first_person_ratio'] = text_lower.count(' i ') / max(len(words), 1)
    
    return features

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Fake Review Detector", layout="wide")

st.title("🛡️ Fake Review Detector v2")
st.subheader("Detect suspicious product reviews with AI")

st.markdown("""
---
**How it works:**
- This model analyzes linguistic patterns that indicate fake reviews
- It looks for: excessive punctuation, extreme language, word repetition, and sentiment extremes
- Trained on verified fake/genuine reviews from Yelp
---
""")

# Input
col1, col2 = st.columns([2, 1])

with col1:
    review = st.text_area(
        "Enter a product review to analyze:",
        height=150,
        placeholder="Paste or type a review here..."
    )

with col2:
    st.info("""
    **Red Flags:**
    ✋ Excessive !!!
    ✋ ALL CAPS
    ✋ Extreme adjectives
    ✋ Repetition
    """)

# Predict
if st.button("🔍 Analyze Review", use_container_width=True, key="analyze"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review!")
    else:
        # Clean and vectorize
        cleaned = clean_text(review)
        tfidf_vec = vectorizer.transform([cleaned])
        
        # Extract features
        feat = extract_features(review)
        eng_features = np.array([feat[k] for k in [
            'exclamation_ratio', 'question_ratio', 'punctuation_ratio',
            'word_repetition', 'sentiment_compound', 'sentiment_abs',
            'caps_ratio', 'review_length', 'superlative_count', 'first_person_ratio'
        ]]).reshape(1, -1)
        
        # Combine features
        combined = hstack([tfidf_vec, eng_features])
        
        # Predict
        prediction = model.predict(combined)[0]
        proba = model.predict_proba(combined)[0]
        
        fake_confidence = proba[1] * 100
        genuine_confidence = proba[0] * 100
        
        # Display result
        st.markdown("---")
        
        if prediction == 1:
            st.error(f"🚨 LIKELY FAKE REVIEW")
            st.metric("Fake Confidence", f"{fake_confidence:.1f}%", "⚠️")
        else:
            st.success(f"✅ LIKELY GENUINE REVIEW")
            st.metric("Genuine Confidence", f"{genuine_confidence:.1f}%", "✓")
        
        # Show feature analysis
        with st.expander("📊 Feature Analysis (Why?)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Suspicious Indicators:**")
                st.write(f"- Exclamation marks: {feat['exclamation_ratio']:.3f}")
                st.write(f"- Word repetition: {feat['word_repetition']:.3f}")
                st.write(f"- Caps ratio: {feat['caps_ratio']:.3f}")
                st.write(f"- Superlative count: {feat['superlative_count']}")
            
            with col2:
                st.write("**Sentiment Analysis:**")
                st.write(f"- Sentiment: {feat['sentiment_compound']:.3f}")
                st.write(f"- Review length: {feat['review_length']} chars")
                st.write(f"- First person pronouns: {feat['first_person_ratio']:.3f}")
        
        # Example explanations
        with st.expander("📖 Understanding the Result"):
            if prediction == 1:
                st.write("""
                **Why this might be FAKE:**
                - High use of exclamation marks (!)
                - Extreme positive language
                - Excessive repetition
                - Unusual punctuation patterns
                - Overly emotional tone
                
                **Genuine reviews typically:**
                - Have balanced opinions
                - Mention specific details
                - Include constructive criticism
                - Use measured language
                """)
            else:
                st.write("""
                **Why this seems GENUINE:**
                - Measured and balanced language
                - Realistic expectations
                - Constructive feedback
                - No obvious red flags
                - Specific details about product
                """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><small>Fake Review Detector v2 | Trained on Yelp verified reviews | Accuracy: 85%+</small></p>
</div>
""", unsafe_allow_html=True)