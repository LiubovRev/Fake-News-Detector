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
## 📂 Project Structure

```text
.
├── app.py                   # Interactive Streamlit web interface
├── fake_news_project.ipynb  # Full pipeline: EDA, Preprocessing, and Model Training
├── models/                  # Local directory for tokenizer & configuration files
│   ├── special_tokens_map.json          
│   └── tokenizer_config.json
│   └── vocab.txt
├── requirements.txt         # List of Python dependencies for deployment
└── README.md                # Project documentation and setup guide
```
---
## 🤖 Model Details

The core of this project is a fine-tuned **DistilBERT** model, optimized for high performance and low latency. The model weights are hosted externally to maintain a lightweight repository.

* **Weights:** [Download from Releases (86MB)](https://github.com/LiubovRev/ML_hometasks/releases/tag/v1.0.0)
* **Architecture:** `distilbert-base-uncased` (Transformer-based)
* **Task:** Binary Sequence Classification
* **Output:** Prediction Label (`FAKE` / `REAL`) with a probability confidence score.
* **Optimization:** Quantized/FP16 precision for faster inference on CPU environments.
* **Preprocessing:** Includes custom logic for **Publisher Leakage Filtering**, ensuring the model generalizes to new data rather than memorizing source signatures.

---

## 🚀 Getting Started

Follow these steps to run the interactive detector on your local machine.

### Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```
---

## Run the app
```bash
streamlit run app.py
```

---
**Notes**
This project is for educational and research purposes. Predictions are probabilistic and not guarantees of factual accuracy.
