import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Heart Disease Predictor ❤️", page_icon="❤️")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "heart_disease_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

st.title("❤️ Heart Disease Prediction")
st.caption("Logistic Regression based classifier")
st.markdown("---")

# User Inputs
age = st.slider("Age", 20, 80, 45)
sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
cp = st.slider("Chest Pain Type (0–3)", 0, 3, 1)
trestbps = st.slider("Resting BP", 90, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.slider("Rest ECG (0–2)", 0, 2, 1)
thalach = st.slider("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
slope = st.slider("Slope (0–2)", 0, 2, 1)
ca = st.slider("Major Vessels (0–4)", 0, 4, 0)
thal = st.slider("Thalassemia (1–3)", 1, 3, 2)

if st.button("🩺 Predict Heart Disease"):
    input_data = np.array([[age, sex, cp, trestbps, chol,
                             fbs, restecg, thalach,
                             exang, oldpeak, slope, ca, thal]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ Heart Disease Detected (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ No Heart Disease Detected (Confidence: {(1-probability)*100:.2f}%)")

st.caption("⚠️ For educational purposes only.")
