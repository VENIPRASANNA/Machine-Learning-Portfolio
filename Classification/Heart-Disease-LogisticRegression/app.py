import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "heart_disease_random_forest.pkl"), "rb"))

# -----------------------------
# Title
# -----------------------------
st.title("❤️ Heart Disease Prediction")
st.caption("Random Forest Classifier")

st.markdown("---")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("🧑 Patient Details")

age = st.slider("Age", 20, 80, 45)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Encode sex
sex_val = 1 if sex == "Male" else 0

# -----------------------------
# Prepare Input Data
# -----------------------------
input_data = pd.DataFrame([[
    age, sex_val, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak,
    slope, ca, thal
]], columns=[
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ No Heart Disease Detected")
