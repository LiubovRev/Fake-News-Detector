# 🌧️ Rain Tomorrow Prediction

Streamlit-додаток для прогнозування ймовірності дощу завтра
на основі погодних умов в Австралії.

## 🔧 Технології
- Python
- Scikit-learn (Random Forest + Pipeline)
- Streamlit

## 🚀 Запуск локально
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Структура проекту

Deployment/  
│  
├── app.py  
├── train_model.ipynb  
├── requirements.txt  
├── README.md  
│  
└── models/  
         └── rain_model.pkl  
└── data/  
         └── weatherAUS.csv 
