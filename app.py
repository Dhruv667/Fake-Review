import streamlit as st
import pickle
import sys

# Load detector
@st.cache_resource
def load_detector():
    try:
        with open('fake_review_detector_v3.pkl', 'rb') as f:
            detector = pickle.load(f)
        return detector
    except:
        st.error("Error: fake_review_detector_v3.pkl not found!")
        st.stop()

detector = load_detector()

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Fake Review Detector v3", layout="wide")

st.title("🛡️ Fake Review Detector v3")
st.subheader("Rule-based detection (No ML - 100% transparent)")

st.markdown("""
**How it works:**
- Analyzes 8 linguistic rules that indicate fake reviews
- Completely transparent - shows WHY it flagged a review
- No machine learning - pure pattern detection
---
""")

# Input
review = st.text_area(
    "Enter a review:",
    height=150,
    placeholder="Paste a review here..."
)

# Analyze
if st.button("🔍 Analyze", use_container_width=True):
    if review.strip() == "":
        st.warning("⚠️ Enter a review!")
    else:
        result = detector.detect_fake(review)
        
        # Result
        st.markdown("---")
        
        if result['is_fake']:
            st.error(f"🚨 LIKELY FAKE REVIEW ({result['confidence']:.1f}% confidence)")
        else:
            st.success(f"✅ LIKELY GENUINE REVIEW ({100-result['confidence']:.1f}% confidence)")
        
        # Reasons
        st.markdown("**Why:**")
        if result['reasons']:
            for reason in result['reasons']:
                st.write(f"• {reason}")
        else:
            st.write("• No obvious red flags detected")
        
        # Details
        with st.expander("📊 Detailed Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Fake Indicators:**")
                feat = result['features']
                st.write(f"- Exclamation marks: {feat['exclamation_count']}")
                st.write(f"- CAPS ratio: {feat['caps_ratio']:.1%}")
                st.write(f"- Superlatives: {feat['superlative_count']}")
                st.write(f"- Word repetition: {feat['word_repetition']:.2f}")
            
            with col2:
                st.write("**Sentiment:**")
                st.write(f"- Overall: {feat['sentiment_compound']:.2f}")
                st.write(f"- Positive: {feat['sentiment_positive']:.1%}")
                st.write(f"- Negative: {feat['sentiment_negative']:.1%}")
                st.write(f"- Review length: {feat['review_length']} chars")

st.markdown("---")
st.markdown("<small>Fake Review Detector v3 | Rule-based | 100% Transparent</small>", unsafe_allow_html=True)