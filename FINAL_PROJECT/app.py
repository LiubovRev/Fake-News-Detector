import streamlit as st
from model_utils import FakeNewsPredictor

st.set_page_config(page_title="GuardianAI: Fake News Detector", page_icon="🛡️")

# Use cache to load the model only once
@st.cache_resource
def get_predictor():
    return FakeNewsPredictor("./models")

st.title("🛡️ GuardianAI")
st.subheader("Deep Learning Based Misinformation Detection")

text_input = st.text_area("Enter news article text here:", height=300, 
                          placeholder="Paste the full text of the article...")

if st.button("Verify Authenticity"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        predictor = get_predictor()
        with st.spinner('Analyzing linguistic patterns and context...'):
            label, confidence = predictor.predict(text_input)
            
        st.divider()
        if label == "FAKE":
            st.error(f"### Result: {label}")
            st.progress(confidence)
            st.write(f"The model is **{confidence:.2%}** certain this is misinformation.")
        else:
            st.success(f"### Result: {label}")
            st.progress(confidence)
            st.write(f"The model is **{confidence:.2%}** certain this is a factual report.")

st.sidebar.markdown("""
### Model Specifications
- **Architecture:** DistilBERT
- **Training F1-Score:** 0.997
- **Inference Speed:** ~200ms
- **Preprocessing:** Anti-Leakage Filter (Reuters/AP)
""")
