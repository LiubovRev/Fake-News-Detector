import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Налаштування сторінки
st.set_page_config(page_title="Weather Predictor", page_icon="🌦️", layout="centered")

# Функція для завантаження моделі та об'єктів препроцесингу
@st.cache_resource
def load_model_objects():
    # Завантажуємо словник, який був збережений у лекції [cite: 6, 7]
    model_data = joblib.load('aussie_rain.joblib')
    return model_data

# Ініціалізація даних
try:
    data = load_model_objects()
    model = data['model']
    scaler = data['scaler']
    encoder = data['encoder']
    input_cols = data['input_cols']
    numeric_cols = data['numeric_cols']
    categorical_cols = data['categorical_cols']
    encoded_cols = data['encoded_cols']
except Exception as e:
    st.error(f"Помилка завантаження моделі: {e}. Переконайтеся, що файл aussie_rain.joblib знаходиться в директорії додатка.")
    st.stop()

st.title("Прогноз дощу в Австралії 🇦🇺🌦️")
st.markdown("""
Цей додаток використовує модель **Random Forest** для визначення ймовірності опадів завтра на основі поточних метеоданих.
""")

# Створення інтерфейсу для введення даних [cite: 31]
st.sidebar.header("Вхідні дані про погоду")

def user_input_features():
    inputs = {}
    
    # Слайдери та поля для числових ознак [cite: 31]
    st.sidebar.subheader("Числові показники")
    inputs['MinTemp'] = st.sidebar.slider("Мінімальна температура (°C)", -5.0, 35.0, 12.0)
    inputs['MaxTemp'] = st.sidebar.slider("Максимальна температура (°C)", 0.0, 50.0, 25.0)
    inputs['Rainfall'] = st.sidebar.number_input("Кількість опадів сьогодні (мм)", 0.0, 300.0, 0.0)
    inputs['Evaporation'] = st.sidebar.number_input("Випаровування (мм)", 0.0, 100.0, 5.0)
    inputs['Sunshine'] = st.sidebar.slider("Сонячні години", 0.0, 15.0, 7.0)
    inputs['WindGustSpeed'] = st.sidebar.slider("Швидкість поривів вітру (км/год)", 0, 130, 40)
    inputs['WindSpeed9am'] = st.sidebar.slider("Швидкість вітру о 9 ранку", 0, 100, 15)
    inputs['WindSpeed3pm'] = st.sidebar.slider("Швидкість вітру о 3 дня", 0, 100, 20)
    inputs['Humidity9am'] = st.sidebar.slider("Вологість о 9 ранку (%)", 0, 100, 60)
    inputs['Humidity3pm'] = st.sidebar.slider("Вологість о 3 дня (%)", 0, 100, 50)
    inputs['Pressure9am'] = st.sidebar.number_input("Тиск о 9 ранку (гПа)", 900.0, 1100.0, 1017.0)
    inputs['Pressure3pm'] = st.sidebar.number_input("Тиск о 3 дня (гПа)", 900.0, 1100.0, 1015.0)
    inputs['Cloud9am'] = st.sidebar.slider("Хмарність о 9 ранку (октанти)", 0, 9, 4)
    inputs['Cloud3pm'] = st.sidebar.slider("Хмарність о 3 дня (октанти)", 0, 9, 4)
    inputs['Temp9am'] = st.sidebar.slider("Температура о 9 ранку (°C)", -5.0, 45.0, 18.0)
    inputs['Temp3pm'] = st.sidebar.slider("Температура о 3 дня (°C)", -5.0, 45.0, 23.0)

    # Випадаючі списки для категоріальних ознак [cite: 31]
    st.sidebar.subheader("Категоріальні показники")
    inputs['Location'] = st.sidebar.selectbox("Локація", ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Canberra', 'Darwin', 'Hobart']) # Список можна розширити
    inputs['WindGustDir'] = st.sidebar.selectbox("Напрям поривів вітру", ['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE', 'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
    inputs['WindDir9am'] = st.sidebar.selectbox("Напрям вітру о 9 ранку", ['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', 'SSW', 'N', 'WSW', 'NW', 'E', 'ESE', 'WNW', 'NNE'])
    inputs['WindDir3pm'] = st.sidebar.selectbox("Напрям вітру о 3 дня", ['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW', 'SW', 'SE', 'N', 'S', 'NNE', 'NE'])
    inputs['RainToday'] = st.sidebar.selectbox("Чи був дощ сьогодні?", ['No', 'Yes'])

    return pd.DataFrame([inputs])

# Отримання вхідних даних
input_df = user_input_features()

st.subheader("Введені дані користувача")
st.write(input_df)

# Кнопка для запуску прогнозу [cite: 35]
if st.button("Зробити прогноз"):
    # ПРЕПРОЦЕСИНГ [cite: 32, 35, 36]
    
    # 1. Масштабування числових ознак [cite: 27, 36]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # 2. Кодування категоріальних ознак [cite: 28, 36]
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    
    # 3. Формування фінального вектора ознак (тільки ті колонки, на яких вчилася модель) [cite: 37]
    X = input_df[input_cols]
    
    # ІНФЕРЕНС (Прогноз) [cite: 37]
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    # Виведення результату [cite: 33, 38, 39]
    st.divider()
    if prediction == 'Yes':
        st.error(f"### Результат: Так, завтра очікується дощ 🌧️")
    else:
        st.success(f"### Результат: Ні, завтра буде сухо ☀️")

    st.write(f"**Ймовірність дощу:** {probability[1]:.2%}")
    st.write(f"**Ймовірність сухої погоди:** {probability[0]:.2%}")
    
    # Додаткова візуалізація (опціонально) [cite: 43]
    st.progress(probability[1])

st.markdown("---")
st.caption("Розроблено в рамках ДЗ: Деплоймент моделі прогнозування погоди.")