# Fake News Detector  


This is a deep learning–based fake news detection system built using a fine-tuned DistilBERT model. It classifies news articles as FAKE or REAL and provides a confidence score through a simple Streamlit web interface.

---
**LIVE-DEMO**
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
