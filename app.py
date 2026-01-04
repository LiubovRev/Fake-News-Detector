import streamlit as st
import torch
import os
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
# Using the direct download links you provided
MODEL_URL = "https://github.com/LiubovRev/ML_hometasks/releases/download/v1.0.0/model.safetensors"
CONFIG_URL = "https://github.com/LiubovRev/ML_hometasks/releases/download/v1.0.0/config.json"
MODEL_DIR = "./models"

st.set_page_config(page_title="Fake News Detector", page_icon="🔍")

def download_file(url, destination):
    if not os.path.exists(destination):
        with st.spinner(f"Downloading {os.path.basename(destination)}..."):
            response = requests.get(url, stream=True)
            response.raise_for_status() # Ensure the download link is valid
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # 1. Download weights and config
    download_file(MODEL_URL, os.path.join(MODEL_DIR, "model.safetensors"))
    download_file(CONFIG_URL, os.path.join(MODEL_DIR, "config.json"))
    
    # 2. Load Tokenizer (Safest to load the base one from HF)
    # This downloads the small vocab files automatically
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # 3. Load Model from the local folder where you downloaded files
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    return tokenizer, model

# --- UI ---
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's Real or Fake using DistilBERT.")

try:
    tokenizer, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

user_input = st.text_area("Article Text:", height=200, placeholder="Paste news content here...")

if st.button("Predict"):
    if user_input.strip():
        # Tokenize
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs).item()
            conf = probs[0][pred].item()

        # Results (Assuming 0: Fake, 1: Real based on common datasets)
        # CHECK: If your training used 0 for Real and 1 for Fake, swap these!
        label = "REAL" if pred == 1 else "FAKE"
        color = "green" if label == "REAL" else "red"

        st.subheader(f"Prediction: :{color}[{label}]")
        st.write(f"Confidence Score: **{conf:.2%}**")
        st.progress(conf)
    else:
        st.warning("Please enter some text first.")
