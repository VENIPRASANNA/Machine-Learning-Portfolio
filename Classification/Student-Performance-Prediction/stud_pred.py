import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Student.csv")

# Target creation
def score_category(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"

df["Performance"] = df["Exam_Score"].apply(score_category)

# ✅ IMPORTANT FEATURES ONLY
features = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Motivation_Level"
]

X = df[features]
y = df["Performance"]

# Encode ONLY Motivation_Level
le = LabelEncoder()
X["Motivation_Level"] = le.fit_transform(X["Motivation_Level"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Accuracy:", accuracy)

# Save files
pickle.dump(model, open("student_dt_model.pkl", "wb"))
pickle.dump(le, open("motivation_encoder.pkl", "wb"))
np.save("accuracy.npy", accuracy)

print("✅ Training completed successfully")
