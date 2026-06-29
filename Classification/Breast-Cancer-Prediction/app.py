import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Cross-Validation Analyzer",
    layout="centered"
)

st.title("🔁 Cross-Validation Analyzer")
st.caption("Stratified K-Fold Evaluation on Cancer Survival Dataset")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Breast_Cancer.csv")

# Encode categorical columns
cat_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Split features & target
X = df.drop("Status", axis=1)
y = df["Status"]

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "KNN", "Random Forest"]
)

k_folds = st.sidebar.slider("Number of Folds (K)", 3, 10, 5)

# -----------------------------
# Model Selection
# -----------------------------
if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "KNN":
    model = KNeighborsClassifier()
else:
    model = RandomForestClassifier(random_state=42)

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", model)
])

# -----------------------------
# Cross Validation
# -----------------------------
skf = StratifiedKFold(
    n_splits=k_folds,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

# -----------------------------
# Display Results
# -----------------------------
st.subheader("📊 Cross-Validation Results")

for i, score in enumerate(scores, 1):
    st.write(f"Fold {i} Accuracy: **{score:.4f}**")

st.success(f"Mean Accuracy: **{scores.mean():.4f}**")

st.markdown("---")
st.caption("Stratified K-Fold preserves class balance across folds.")
