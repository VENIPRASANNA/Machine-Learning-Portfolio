import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Fitness Anomaly Detection using DBSCAN",
    page_icon="🏃",
    layout="centered"
)

st.title("🏃 Fitness Anomaly Detection using DBSCAN")
st.caption("Unsupervised anomaly detection on fitness activity data")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("fitness.csv")

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Feature Selection (NUMERIC ONLY)
# --------------------------------------------------
features = [
    "duration_minutes",
    "calories_burned",
    "avg_heart_rate",
    "daily_steps",
    "bmi"
]

df_clean = df[features].dropna()

# --------------------------------------------------
# Standardization
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# --------------------------------------------------
# EPS Selection (K-Distance Graph)
# --------------------------------------------------
st.subheader("📐 EPS Selection (K-Distance Graph)")

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

distances = np.sort(distances[:, 4])

plt.figure(figsize=(7,4))
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("5-NN Distance")
plt.title("K-Distance Graph")
st.pyplot(plt)

# --------------------------------------------------
# DBSCAN Parameters
# --------------------------------------------------
st.subheader("⚙ DBSCAN Parameters")

eps = st.slider("Select EPS value", 0.5, 3.0, 1.5, 0.1)
min_samples = st.slider("Select Min Samples", 3, 10, 5)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(X_scaled)

df_clean["Cluster"] = clusters

# --------------------------------------------------
# Results
# --------------------------------------------------
st.subheader("🔍 Clustering Summary")

normal = (df_clean["Cluster"] != -1).sum()
anomalies = (df_clean["Cluster"] == -1).sum()

col1, col2 = st.columns(2)
col1.metric("✅ Normal Records", normal)
col2.metric("🚨 Anomalies Detected", anomalies)

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.subheader("📈 Cluster Visualization")

plt.figure(figsize=(7,5))
plt.scatter(
    df_clean["calories_burned"],
    df_clean["avg_heart_rate"],
    c=df_clean["Cluster"],
    cmap="viridis",
    s=60
)

plt.xlabel("Calories Burned")
plt.ylabel("Average Heart Rate")
plt.title("DBSCAN Clustering Result")
st.pyplot(plt)

# --------------------------------------------------
# Show Anomalies
# --------------------------------------------------
st.subheader("🚨 Detected Anomalies")
st.dataframe(df_clean[df_clean["Cluster"] == -1])

# --------------------------------------------------
# Explanation
# --------------------------------------------------
st.markdown("""
### 🧠 Explanation
- DBSCAN is a **density-based clustering algorithm**
- Points labeled **-1** are considered anomalies
- Data is **standardized** because DBSCAN uses distance
- This method is effective for **unsupervised anomaly detection**
""")
