
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load training and testing data
train_data = pd.read_csv("outputs/train_data.csv")
test_data = pd.read_csv("outputs/test_data.csv")

# Separate features and target
X_train = train_data.drop("Leakage_Flag", axis=1)
y_train = train_data["Leakage_Flag"]

X_test = test_data.drop("Leakage_Flag", axis=1)
y_test = test_data["Leakage_Flag"]

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("MODEL TRAINING COMPLETED!")
print("\nAccuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "models/water_leakage_model.pkl")

print("\nModel saved successfully!")

from sklearn.model_selection import cross_val_score, StratifiedKFold

print("\nCROSS VALIDATION")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("Cross Validation Scores:", scores)
print("Average CV Accuracy:", scores.mean() * 100, "%")
