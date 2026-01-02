# 🌧️ Rain Tomorrow Prediction: End-to-End ML Pipeline

### [🚀 Live Demo on Streamlit](https://mlhometasks-hpttpgdx7cjxsnwemqrz7b.streamlit.app/)

This repository features a complete Machine Learning workflow, from data exploration and model training to deploying a web-based application. The project predicts the probability of rainfall in Australia for the following day based on historical meteorological data.

---

## 📋 Project Overview
The goal of this project is to build a robust classification model to handle imbalanced weather data and provide real-time predictions via a user-friendly interface.

### Key Features:
* **Data Preprocessing:** Robust handling of missing values and categorical encoding using **Scikit-learn Pipelines**.
* **Model:** **Random Forest Classifier** optimized for binary classification performance.
* **Deployment:** Interactive web application built with **Streamlit** and served via Streamlit Cloud.
* **Reproducibility:** Serialized model using `pickle` to ensure consistent inference across environments.

---

## 🛠️ Tech Stack
* **Language:** Python
* **ML Framework:** Scikit-learn (Random Forest, Pipeline, Imputers)
* **Data Analysis:** Pandas, NumPy
* **Deployment:** Streamlit

---

## 📂 Project Structure
```plaintext
Deployment/  
│  
├── app.py                # Streamlit application script
├── train_model.ipynb     # Jupyter Notebook with EDA and model training
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
│  
├── models/  
│   └── rain_model.pkl    # Serialized Random Forest model
└── data/  
    └── weatherAUS.csv    # Dataset source (Kaggle: Weather in Australia)
```
---

## 🚀 Local Setup

To run this project locally, follow these steps:

Clone the repository:
```
git clone [https://github.com/LiubovRev/Rain-Tomorrow-Prediction.git](https://github.com/LiubovRev/Rain-Tomorrow-Prediction.git)
cd Rain-Tomorrow-Prediction/Deployment
```

Install dependencies:
```
pip install -r requirements.txt
```

Launch the App:
```
 streamlit run app.py
```

## 📊 Methodology & Performance

In the `train_model.ipynb`, the model was evaluated using precision-recall curves and F1-scores to ensure reliability despite the class imbalance typical of rainfall datasets. The use of a Pipeline ensures that the preprocessing steps applied to the training data are identical to those applied to user inputs in the production app.
