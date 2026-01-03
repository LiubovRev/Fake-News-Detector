# Fake News Detector  


This is a deep learning–based fake news detection system built using a fine-tuned DistilBERT model. It classifies news articles as FAKE or REAL and provides a confidence score through a simple Streamlit web interface.

---
[STREAMLIT LIVE-DEMO](https://fwssuuuutqxmdjczkeasw8.streamlit.app/) 
---

## 📊 Dataset Overview

The dataset is a comprehensive collection of real and fake news articles used to train and validate the detection models. It consists of over **40,000 labeled records**, providing a balanced foundation for linguistic analysis.

### **Data Specifications**
* **Source:** [Download from Google Drive](https://drive.google.com/file/d/16BERzRTy-EKFcJ-WuEVlzVkSWnyQkafQ/view?usp=sharing)
* **Format:** Structured CSV
* **Labels:** * `0`: **Fake News** (Fabricated or misleading content)
    * `1`: **Real News** (Verified factual reporting)

### **Key Features**
| Feature | Description |
| :--- | :--- |
| **title** | The headline of the article (analyzed for capitalization ratios). |
| **text** | The full body content (primary input for DistilBERT). |
| **subject** | News category (Politics, World News, Government, etc.). |
| **date** | Original publication timestamp. |

### **Pre-processing Highlights**
To ensure the model learned semantic patterns rather than "cheating" through metadata, the following steps were taken:

1. **Leakage Removal:** Stripped source-specific markers (e.g., "Reuters") and location tags that were present primarily in real news.
2. **Text Normalization:** Lowercasing, removal of special characters, and whitespace optimization.
3. **Feature Engineering:** Created a `title_caps_ratio` feature to capture the tendency of fake news to use "clickbait" capitalization.
---

## Features

- Binary fake news classification
- DistilBERT transformer model 
- High accuracy (F1 ≈ 0.997)
- Fast inference (~200ms)
- Streamlit-based web UI

---

## Project Structure

├── app.py # Streamlit application    
├── fake_news_project.ipynb # Model training and evaluation    
├── models/ # Saved model and tokenizer  
├── data/  
└── README.md  
---

## Model

The trained DistilBERT model (86MB) can be downloaded from the [Releases section](https://github.com/LiubovRev/ML_hometasks/releases/tag/v1.0.0).

- **Architecture:** DistilBERT
- **Task:** Text classification
- **Output:** Label (FAKE / REAL) + confidence score
- **Preprocessing:** Publisher leakage filtering

---

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run app.py
```

## Notes

This project is for educational and research purposes. Predictions are probabilistic and not guarantees of factual accuracy.
