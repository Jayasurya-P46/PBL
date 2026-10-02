import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("../dataset/gesture_dataset_v2.csv")
# -------------------------
# Split Features and Labels
# -------------------------
X = df.drop("label", axis=1)
y = df["label"]

# -------------------------
# Train-Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# Train Random Forest
# -------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------
# Predict
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

# -------------------------
# Save Model
# -------------------------
joblib.dump(model, "../models/gesture_rf_v2.pkl")

print("\nModel saved as models/gesture_rf_v2.pkl")
