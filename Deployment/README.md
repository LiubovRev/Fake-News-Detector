# 🌧️ Rain Tomorrow Prediction 

====== https://mlhometasks-hpttpgdx7cjxsnwemqrz7b.streamlit.app/ ======

Streamlit-додаток для прогнозування ймовірності дощу завтра
на основі погодних умов в Австралії.

<img width="2476" height="1058" alt="зображення" src="https://github.com/user-attachments/assets/9f9bf063-544e-4339-9102-c5333f2e222e" />


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
