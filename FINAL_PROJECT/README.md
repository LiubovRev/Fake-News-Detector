# Fake News Detector  


This is a deep learning–based fake news detection system built using a fine-tuned DistilBERT model. It classifies news articles as FAKE or REAL and provides a confidence score through a simple Streamlit web interface.

---
[STREAMLIT LIVE-DEMO](https://fwssuuuutqxmdjczkeasw8.streamlit.app/) 
---

## Dataset
The dataset used for this project is a collection of real and fake news articles. It contains thousands of records where each entry includes the article title, text, and date.

- **Source:** [Download from Google Drive](https://drive.google.com/file/d/16BERzRTy-EKFcJ-WuEVlzVkSWnyQkafQ/view?usp=sharing)
- **Format:** CSV
- **Features:**
  - `title`: The headline of the article.
  - `text`: The full body of the news story.
  - `subject`: The category of the news (e.g., Politics, World News).
  - `date`: Publication date.
- **Target:** - `0`: Fake News
  - `1`: Real News
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
