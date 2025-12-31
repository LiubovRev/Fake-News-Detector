import streamlit as st
import joblib
import pandas as pd
import os

# Налаштування сторінки
st.set_page_config(page_title="Weather Predictor", page_icon="🌦️", layout="centered")

# Функція для завантаження моделі
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'models', 'rain_model.pkl')
    pipeline = joblib.load(model_path)
    return pipeline

# Ініціалізація моделі
try:
    pipeline = load_model()
    st.success("✓ Модель успішно завантажена!")
except Exception as e:
    st.error(f"Помилка завантаження моделі: {e}")
    st.stop()

st.title("Прогноз дощу в Австралії 🇦🇺🌦️")
st.markdown("""
Цей додаток використовує модель **Logistic Regression** для визначення ймовірності опадів завтра на основі поточних метеоданих.
""")

# Створення інтерфейсу для введення даних
st.sidebar.header("Вхідні дані про погоду")

def user_input_features():
    inputs = {}
    
    # Слайдери та поля для числових ознак
    st.sidebar.subheader("Числові показники")
    inputs['MinTemp'] = st.sidebar.slider("Мінімальна температура (°C)", -10.0, 40.0, 12.0)
    inputs['MaxTemp'] = st.sidebar.slider("Максимальна температура (°C)", -5.0, 50.0, 25.0)
    inputs['Rainfall'] = st.sidebar.number_input("Кількість опадів сьогодні (мм)", 0.0, 300.0, 0.0)
    inputs['Evaporation'] = st.sidebar.number_input("Випаровування (мм)", 0.0, 150.0, 5.0)
    inputs['Sunshine'] = st.sidebar.slider("Сонячні години", 0.0, 15.0, 7.0)
    inputs['WindGustSpeed'] = st.sidebar.slider("Швидкість поривів вітру (км/год)", 0, 150, 40)
    inputs['WindSpeed9am'] = st.sidebar.slider("Швидкість вітру о 9 ранку (км/год)", 0, 130, 15)
    inputs['WindSpeed3pm'] = st.sidebar.slider("Швидкість вітру о 3 дня (км/год)", 0, 130, 20)
    inputs['Humidity9am'] = st.sidebar.slider("Вологість о 9 ранку (%)", 0, 100, 60)
    inputs['Humidity3pm'] = st.sidebar.slider("Вологість о 3 дня (%)", 0, 100, 50)
    inputs['Pressure9am'] = st.sidebar.number_input("Тиск о 9 ранку (гПа)", 900.0, 1100.0, 1017.0)
    inputs['Pressure3pm'] = st.sidebar.number_input("Тиск о 3 дня (гПа)", 900.0, 1100.0, 1015.0)
    inputs['Cloud9am'] = st.sidebar.slider("Хмарність о 9 ранку (октанти)", 0, 9, 4)
    inputs['Cloud3pm'] = st.sidebar.slider("Хмарність о 3 дня (октанти)", 0, 9, 4)
    inputs['Temp9am'] = st.sidebar.slider("Температура о 9 ранку (°C)", -10.0, 45.0, 18.0)
    inputs['Temp3pm'] = st.sidebar.slider("Температура о 3 дня (°C)", -10.0, 45.0, 23.0)

    # Випадаючі списки для категоріальних ознак
    st.sidebar.subheader("Категоріальні показники")
    
    # Список всіх локацій з датасету weatherAUS
    locations = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
                 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
                 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown',
                 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat',
                 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura',
                 'Nhil', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
                 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
                 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
                 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
                 'AliceSprings', 'Darwin', 'Katherine', 'Uluru']
    
    inputs['Location'] = st.sidebar.selectbox("Локація", sorted(locations))
    
    wind_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    inputs['WindGustDir'] = st.sidebar.selectbox("Напрям поривів вітру", wind_directions)
    inputs['WindDir9am'] = st.sidebar.selectbox("Напрям вітру о 9 ранку", wind_directions)
    inputs['WindDir3pm'] = st.sidebar.selectbox("Напрям вітру о 3 дня", wind_directions)
    
    inputs['RainToday'] = st.sidebar.selectbox("Чи був дощ сьогодні?", [0, 1], format_func=lambda x: "Так" if x == 1 else "Ні")

    return pd.DataFrame([inputs])

# Отримання вхідних даних
input_df = user_input_features()

st.subheader("Введені дані користувача")
# Відображення з читабельними значеннями
display_df = input_df.copy()
display_df['RainToday'] = display_df['RainToday'].map({0: 'Ні', 1: 'Так'})
st.write(display_df)

# Кнопка для запуску прогнозу
if st.button("🌦️ Зробити прогноз", type="primary"):
    try:
        # Pipeline сам виконає всю preprocessingu!
        # Не потрібно окремо масштабувати чи кодувати - pipeline все зробить
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0]
        
        # Виведення результату
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("### 🌧️ ТАК")
                st.write("Завтра очікується дощ")
            else:
                st.success("### ☀️ НІ")
                st.write("Завтра буде сухо")
        
        with col2:
            st.metric("Ймовірність дощу", f"{probability[1]:.1%}")
            st.metric("Ймовірність без дощу", f"{probability[0]:.1%}")
        
        # Прогрес-бар
        st.write("### Візуалізація ймовірності")
        st.progress(float(probability[1]))
        
        # Інтерпретація результату
        st.divider()
        st.write("### 💡 Інтерпретація")
        if probability[1] > 0.7:
            st.info("🌧️ Висока ймовірність дощу. Краще взяти парасольку!")
        elif probability[1] > 0.4:
            st.warning("⛅ Помірна ймовірність дощу. Можливо, варто підготуватися.")
        else:
            st.success("☀️ Низька ймовірність дощу. Скоріш за все буде гарна погода!")
        
    except Exception as e:
        st.error(f"❌ Помилка під час прогнозу: {e}")
        st.exception(e)

st.markdown("---")
st.caption("Розроблено в рамках ДЗ: Деплоймент моделі прогнозування погоди.")