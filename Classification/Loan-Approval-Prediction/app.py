import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Loan Default Prediction 💳")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR,"credit_knn_model.pkl"),"rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR,"scaler.pkl"),"rb"))

st.title("💳 Loan Default Prediction")
st.caption("KNN Classification Model")

age = st.slider("Age", 18, 70, 35)
ed = st.slider("Education Level", 1, 5, 3)
employ = st.slider("Years Employed", 0, 40, 5)
address = st.slider("Years at Current Address", 0, 30, 5)
income = st.number_input("Annual Income", min_value=0)
debtinc = st.slider("Debt-Income Ratio", 0.0, 50.0, 20.0)
creddebt = st.number_input("Credit Card Debt", min_value=0.0)
othdebt = st.number_input("Other Debt", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[age,ed,employ,address,income,
                             debtinc,creddebt,othdebt]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High Risk: Loan cannot be approved")
    else:
        st.success("✅ Low Risk: Loan can be approved")
