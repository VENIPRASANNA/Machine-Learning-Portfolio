import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "student_dt_model.pkl"), "rb"))
le = pickle.load(open(os.path.join(BASE_DIR, "motivation_encoder.pkl"), "rb"))
accuracy = np.load(os.path.join(BASE_DIR, "accuracy.npy"))

st.set_page_config(page_title="Student Performance Predictor 🎓")

st.title("🎓 Student Performance Prediction")
st.markdown(f"**Model Accuracy:** {accuracy:.2f}")
st.markdown("---")

# Inputs
hours = st.slider("Hours Studied", 0, 12, 5)
attendance = st.slider("Attendance (%)", 50, 100, 80)
sleep = st.slider("Sleep Hours", 4, 10, 7)
previous = st.slider("Previous Scores", 0, 100, 60)
tutoring = st.slider("Tutoring Sessions", 0, 10, 2)
motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"])

# Prepare input
input_df = pd.DataFrame([{
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Sleep_Hours": sleep,
    "Previous_Scores": previous,
    "Tutoring_Sessions": tutoring,
    "Motivation_Level": le.transform([motivation])[0]
}])

# Predict
if st.button("Predict Performance"):
    pred = model.predict(input_df)[0]

    if pred == "High":
        st.success("🏆 HIGH Performance")
    elif pred == "Medium":
        st.warning("📘 MEDIUM Performance")
    else:
        st.error("📕 LOW Performance")
