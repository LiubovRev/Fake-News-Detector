import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
import os

# --- CONFIGURATION ---
# Using the direct download links you provided
MODEL_URL = "https://github.com/LiubovRev/Fake-News-Detector/releases/download/v3/model.safetensors"
CONFIG_URL = "https://github.com/LiubovRev/Fake-News-Detector/releases/download/v3/config.json"
MODEL_DIR = "./models"

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .warning-box {
        padding: 1rem;
        background-color: #FFF3CD;
        border-left: 4px solid #FFC107;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

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


# Text preprocessing
def clean_text(text):
    """Basic text cleaning for BERT input."""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

# Prediction function
def predict_news(text, tokenizer, model, device):
    """Make prediction on input text."""
    if not text.strip():
        return None, None, None
    
    # Clean and tokenize
    cleaned_text = clean_text(text)
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
        
    return prediction, confidence, probs[0].cpu().numpy()

# Feature extraction for interpretation
def extract_features(text):
    """Extract interpretable features from text."""
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'has_reuters': 1 if 'reuters' in text.lower() else 0,
        'has_source': 1 if any(s in text.lower() for s in ['reuters', 'ap', 'afp', 'bloomberg']) else 0
    }
    return features

# Main app
def main():
    # Load model
    tokenizer, model, device = load_model()
    
    # Header
    st.markdown('<div class="main-header">📰 Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered News Article Classification with DistilBERT</div>', unsafe_allow_html=True)
    
    # Warning banner
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Important Disclaimer:</strong> This model is for educational and research purposes only. 
        It achieves 99.99% accuracy on training data but shows significant limitations on real-world data. 
        Always verify information through multiple credible sources.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model info
        st.subheader("Model Information")
        st.write("**Architecture:** DistilBERT")
        st.write("**Parameters:** 66M")
        st.write("**Training Accuracy:** 99.99%")
        
        # Device info
        if model is not None:
            device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            st.write(f"**Device:** {device_name}")
        
        st.markdown("---")
        
        # Confidence threshold
        st.subheader("Confidence Threshold")
        confidence_threshold = st.slider(
            "Minimum confidence for classification",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Predictions below this threshold will be flagged for review"
        )
        
        st.markdown("---")
        
        # About
        st.subheader("📚 About")
        st.write("""
        This model was trained on 38,227 news articles to classify them as real or fake.
        
        **Known Limitations:**
        - Biased toward 2017 articles
        - Vulnerable to source injection
        - May not generalize to short statements
        - Overconfident predictions
        """)
        
        st.markdown("---")
        
        # Example articles
        st.subheader("📝 Example Articles")
        if st.button("Load Fake News Example"):
            st.session_state['example_text'] = """
            BREAKING: SHOCKING New Evidence Reveals Government Cover-Up!!!
            
            Multiple sources confirm that documents leaked today expose massive corruption at the highest levels. 
            The mainstream media REFUSES to report this EXPLOSIVE information that will change EVERYTHING we know!
            
            SHARE THIS BEFORE IT GETS DELETED!!! The truth is finally coming out and they don't want you to know!!!
            """
        
        if st.button("Load Real News Example"):
            st.session_state['example_text'] = """
            WASHINGTON (Reuters) - The Federal Reserve announced on Tuesday that it would maintain 
            interest rates at their current level, citing ongoing concerns about inflation and 
            economic growth.
            
            In a statement following its two-day policy meeting, the Fed said it would continue 
            to monitor economic indicators closely. Chair Jerome Powell is expected to hold a 
            press conference later today to discuss the decision.
            
            Economists surveyed by Reuters had largely expected the Fed to hold rates steady, 
            with most projecting no change until the fourth quarter.
            """
    
    # Main content area
    if model is None:
        st.error("Model could not be loaded. Please check the model directory and try again.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Article", "📊 Model Performance", "🛡️ Robustness Tests"])
    
    with tab1:
        st.header("Analyze News Article")
        
        # Text input
        default_text = st.session_state.get('example_text', '')
        article_text = st.text_area(
            "Paste your news article here:",
            value=default_text,
            height=300,
            placeholder="Enter or paste a news article to analyze..."
        )
        
        # Analysis columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            analyze_button = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
        
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
            if clear_button:
                st.session_state['example_text'] = ''
                st.rerun()
        
        # Perform analysis
        if analyze_button and article_text.strip():
            with st.spinner("Analyzing article..."):
                prediction, confidence, probs = predict_news(article_text, tokenizer, model, device)
                
                if prediction is not None:
                    # Results section
                    st.markdown("---")
                    st.subheader("Analysis Results")
                    
                    # Prediction cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        label = "🔴 FAKE NEWS" if prediction == 1 else "✅ REAL NEWS"
                        color = "#E57373" if prediction == 1 else "#7BC86C"
                        st.markdown(f"""
                        <div class="metric-card" style="background-color: {color};">
                            <h2 style="color: white; margin: 0;">{label}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                        if confidence < confidence_threshold:
                            st.warning("⚠️ Low confidence - requires review")
                    
                    with col3:
                        certainty = "High" if confidence > 0.95 else "Medium" if confidence > 0.8 else "Low"
                        st.metric("Certainty Level", certainty)
                    
                    # Probability distribution
                    st.markdown("---")
                    st.subheader("Prediction Breakdown")
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Real News', 'Fake News'],
                            y=[probs[0], probs[1]],
                            marker_color=['#7BC86C', '#E57373'],
                            text=[f'{probs[0]*100:.1f}%', f'{probs[1]*100:.1f}%'],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Classification Probabilities",
                        yaxis_title="Probability",
                        yaxis=dict(range=[0, 1]),
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature analysis
                    st.markdown("---")
                    st.subheader("Article Features")
                    
                    features = extract_features(article_text)
                    
                    feat_col1, feat_col2, feat_col3 = st.columns(3)
                    with feat_col1:
                        st.metric("Word Count", features['word_count'])
                        st.metric("Character Count", features['length'])
                    with feat_col2:
                        st.metric("Caps Ratio", f"{features['caps_ratio']*100:.1f}%")
                        st.metric("Exclamations", features['exclamation_count'])
                    with feat_col3:
                        st.metric("Questions", features['question_count'])
                        st.metric("Has Source Citation", "Yes" if features['has_source'] else "No")
                    
                    # Interpretation
                    st.markdown("---")
                    st.subheader("💡 Interpretation")
                    
                    if features['caps_ratio'] > 0.15:
                        st.warning("⚠️ High capitalization ratio detected - common in fake news for clickbait")
                    if features['exclamation_count'] > 5:
                        st.warning("⚠️ Excessive exclamation marks - may indicate emotional manipulation")
                    if features['has_source'] == 0:
                        st.info("ℹ️ No credible source citation detected")
                    if confidence < confidence_threshold:
                        st.error("⚠️ Model is uncertain about this classification. Human review recommended.")
    
    with tab2:
        st.header("Model Performance Metrics")
        
        # Performance table
        st.subheader("Model Comparison")
        
        performance_data = {
            'Model': ['Logistic Regression', 'Custom Ensemble', 'XGBoost', 'DistilBERT'],
            'Accuracy': [0.985, 0.985, 0.997, 0.9999],
            'F1-Score': [0.984, 0.984, 0.997, 0.9999],
            'Training Time': ['< 1 min', '~2 min', '~5 min', '~30 min']
        }
        
        df_performance = pd.DataFrame(performance_data)
        
        # Highlight best model
        st.dataframe(
            df_performance.style.highlight_max(subset=['Accuracy', 'F1-Score'], color='lightgreen'),
            use_container_width=True
        )
        
        # Performance visualization
        fig = px.bar(
            df_performance,
            x='Model',
            y=['Accuracy', 'F1-Score'],
            title='Model Performance Comparison',
            barmode='group',
            color_discrete_map={'Accuracy': '#4A90E2', 'F1-Score': '#F28B82'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix (simulated for demo)
        st.markdown("---")
        st.subheader("DistilBERT Confusion Matrix")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            confusion_data = pd.DataFrame(
                [[4164, 0], [1, 3488]],
                columns=['Predicted Real', 'Predicted Fake'],
                index=['Actual Real', 'Actual Fake']
            )
            st.dataframe(confusion_data, use_container_width=True)
        
        with col2:
            st.metric("True Positives", "3,488")
            st.metric("True Negatives", "4,164")
            st.metric("False Positives", "0")
            st.metric("False Negatives", "1")
    
    with tab3:
        st.header("Robustness Testing Results")
        
        st.warning("""
        ⚠️ **Critical Finding:** While the model achieves 99.99% accuracy on test data, 
        it shows significant vulnerabilities when tested against adversarial conditions and external datasets.
        """)
        
        # Robustness metrics
        robustness_data = {
            'Test Type': [
                'In-Domain (Test Set)',
                'External Dataset (LIAR)',
                'Source Injection Attack',
                'Character Noise (10%)',
                'Synonym Substitution (95%)'
            ],
            'Accuracy': [99.99, 43.7, 38.0, 99.0, 99.99],
            'Status': ['✅ Pass', '❌ Fail', '❌ Fail', '⚠️ Overconfident', '⚠️ Overconfident']
        }
        
        df_robustness = pd.DataFrame(robustness_data)
        st.dataframe(df_robustness, use_container_width=True)
        
        # Visualization
        fig = go.Figure(data=[
            go.Bar(
                x=df_robustness['Test Type'],
                y=df_robustness['Accuracy'],
                marker_color=['#7BC86C', '#E57373', '#E57373', '#F6C26B', '#F6C26B'],
                text=df_robustness['Accuracy'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto',
            )
        ])
        fig.update_layout(
            title="Robustness Across Different Test Conditions",
            yaxis_title="Accuracy (%)",
            xaxis_title="Test Type",
            height=400,
            yaxis=dict(range=[0, 105])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed findings
        st.markdown("---")
        st.subheader("Detailed Findings")
        
        with st.expander("📉 External Dataset Collapse"):
            st.write("""
            **Problem:** Accuracy dropped from 99.99% to 43.7% on the LIAR dataset.
            
            **Cause:** Model learned dataset-specific patterns (article length, formatting) 
            rather than transferable deception signals.
            
            **Impact:** Cannot reliably classify short political statements.
            """)
        
        with st.expander("🎯 Source Injection Vulnerability"):
            st.write("""
            **Problem:** 62% of fake news was misclassified as real after adding "WASHINGTON (Reuters) -"
            
            **Cause:** Model relies on formatting patterns despite explicit source removal during training.
            
            **Impact:** Attackers can easily fool the model by mimicking credible source formatting.
            """)
        
        with st.expander("🔒 Overconfidence on Perturbed Input"):
            st.write("""
            **Problem:** Model maintains >99% confidence even with 10% character corruption.
            
            **Cause:** Reliance on few highly predictive tokens rather than distributed understanding.
            
            **Impact:** False sense of robustness; model doesn't recognize when it should be uncertain.
            """)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        st.info("""
        **For Production Use:**
        1. Always use confidence thresholding (flag predictions below 80%)
        2. Implement human-in-the-loop for borderline cases
        3. Test on domain-specific data before deployment
        4. Monitor for adversarial attacks and distribution shift
        5. Combine with fact-checking APIs and human expertise
        """)

if __name__ == "__main__":
    main()