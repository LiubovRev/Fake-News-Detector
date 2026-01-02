🌧️ Rain Tomorrow Prediction: End-to-End ML Pipeline
🚀 Live Demo on Streamlit

This repository features a complete Machine Learning workflow, from data exploration and model training to deploying a web-based application. The project predicts the probability of rainfall in Australia for the following day based on historical meteorological data.
📋 Project Overview

The goal of this project is to build a robust classification model to handle imbalanced weather data and provide real-time predictions via a user-friendly interface.
Key Features:

    Data Preprocessing: Handled missing values and categorical encoding using Scikit-learn Pipelines.

    Model: Random Forest Classifier optimized for binary classification.

    Deployment: Interactive web app built with Streamlit and served via Streamlit Cloud.

    Reproducibility: Serialized model using pickle for consistent inference.

🛠️ Tech Stack

    Language: Python

    ML Framework: Scikit-learn (Random Forest, Pipeline, Imputers)

    Data Analysis: Pandas, NumPy

    Deployment: Streamlit

📂 Project Structure
Plaintext

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

🚀 Local Setup

To run this project locally, follow these steps:

    Clone the repository:
    Bash

git clone https://github.com/LiubovRev/Rain-Tomorrow-Prediction.git
cd Rain-Tomorrow-Prediction/Deployment

Install dependencies:
Bash

pip install -r requirements.txt

Launch the App:
Bash

    streamlit run app.py

📊 Model Performance

In the train_model.ipynb, the model was evaluated using accuracy, precision, and recall to ensure reliability despite the class imbalance typical of rainfall datasets.
