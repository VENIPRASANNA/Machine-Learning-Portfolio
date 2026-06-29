import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# Page Config (MUST be first)
# -----------------------------
st.set_page_config(
    page_title="Bank Customer Segmentation",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------
# App Header (RENDER IMMEDIATELY)
# -----------------------------
st.title("🏦 Bank Customer Segmentation")
st.caption("K-Means Clustering (Unsupervised Learning)")
st.markdown("---")

# -----------------------------
# Cached training function
# -----------------------------
@st.cache_resource
def train_kmeans():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, "bank_churn.csv"))

    features = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary"
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    return kmeans, scaler

# -----------------------------
# Train model (cached)
# -----------------------------
with st.spinner("Training K-Means model..."):
    kmeans, scaler = train_kmeans()

st.success("✅ Model ready")

# -----------------------------
# User Inputs
# -----------------------------
credit_score = st.slider("Credit Score", 300, 900, 650)
age = st.slider("Age", 18, 80, 40)
tenure = st.slider("Tenure (years)", 0, 10, 5)
balance = st.number_input("Account Balance", value=50000.0)
products = st.slider("Number of Products", 1, 4, 2)
salary = st.number_input("Estimated Salary", value=60000.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Find Customer Cluster"):
    input_data = np.array([[credit_score, age, tenure, balance, products, salary]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"🧠 Customer belongs to **Cluster {cluster}**")
