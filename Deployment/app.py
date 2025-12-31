import streamlit as st
import pandas as pd
import joblib

# ======================
# Page config
# ======================
st.set_page_config(
    page_title="Rain Tomorrow Prediction",
    page_icon="🌧️",
    layout="centered"
)

st.title("🌧️ Прогноз дощу на завтра")
st.write(
    "Додаток використовує модель Random Forest "
    "з повним preprocessing пайплайном."
)

# ======================
# Load model
# ======================
@st.cache_resource
def load_pipeline():
    return joblib.load("models/aussie_rain.joblib")

pipeline = load_pipeline()

# ======================
# User input form
# ======================
st.subheader("🔧 Вхідні дані")

with st.form("weather_form"):
    col1, col2 = st.columns(2)

    with col1:
        MinTemp = st.number_input("MinTemp (°C)", -10.0, 40.0, 10.0)
        MaxTemp = st.number_input("MaxTemp (°C)", -10.0, 50.0, 25.0)
        Rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 0.0)
        WindGustSpeed = st.number_input("WindGustSpeed (km/h)", 0.0, 150.0, 30.0)
        Humidity9am = st.number_input("Humidity9am (%)", 0.0, 100.0, 60.0)

    with col2:
        Humidity3pm = st.number_input("Humidity3pm (%)", 0.0, 100.0, 50.0)
        Pressure9am = st.number_input("Pressure9am (hPa)", 980.0, 1040.0, 1010.0)
        Pressure3pm = st.number_input("Pressure3pm (hPa)", 980.0, 1040.0, 1008.0)
        Temp9am = st.number_input("Temp9am (°C)", -10.0, 40.0, 15.0)
        Temp3pm = st.number_input("Temp3pm (°C)", -10.0, 45.0, 22.0)

    RainToday = st.selectbox("Чи йшов дощ сьогодні?", ["No", "Yes"])
    Location = st.selectbox(
        "Локація",
        [
            "Sydney", "Melbourne", "Brisbane",
            "Perth", "Adelaide"
        ]
    )

    submitted = st.form_submit_button("🔮 Спрогнозувати")

# ======================
# Prediction
# ======================
if submitted:
    input_df = pd.DataFrame([{
        "MinTemp": MinTemp,
        "MaxTemp": MaxTemp,
        "Rainfall": Rainfall,
        "WindGustSpeed": WindGustSpeed,
        "Humidity9am": Humidity9am,
        "Humidity3pm": Humidity3pm,
        "Pressure9am": Pressure9am,
        "Pressure3pm": Pressure3pm,
        "Temp9am": Temp9am,
        "Temp3pm": Temp3pm,
        "RainToday": RainToday,
        "Location": Location
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    st.subheader("📊 Результат прогнозу")

    if prediction == 1:
        st.error(
            f"🌧️ **Завтра ОЧІКУЄТЬСЯ дощ**\n\n"
            f"Ймовірність: **{probability:.2%}**"
        )
    else:
        st.success(
            f"☀️ **Завтра дощ НЕ очікується**\n\n"
            f"Ймовірність: **{probability:.2%}**"
        )
