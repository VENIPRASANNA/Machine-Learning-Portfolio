import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Page config
st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

st.title("🛍️ Mall Customer Segmentation")
st.caption("Hierarchical Clustering using Age, Income & Spending Score")

# Load data
df = pd.read_csv("Mall_Customers.csv")

# Feature selection
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar
st.sidebar.header("⚙️ Settings")
k = st.sidebar.slider("Number of Clusters", 2, 6, 4)

# Dendrogram
st.subheader("🌳 Dendrogram")
linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(8, 4))
dendrogram(linked)
plt.xlabel("Customers")
plt.ylabel("Distance")
st.pyplot(plt)

# Clustering
model = AgglomerativeClustering(n_clusters=k)
labels = model.fit_predict(X_scaled)

df["Cluster"] = labels

# Display results
st.subheader("📊 Clustered Customers")
st.dataframe(df)

# Visualization
st.subheader("📈 Cluster Visualization")

plt.figure(figsize=(6, 4))
plt.scatter(
    X_scaled[:, 1],
    X_scaled[:, 2],
    c=labels
)
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title(f"Clusters (k = {k})")
st.pyplot(plt)
