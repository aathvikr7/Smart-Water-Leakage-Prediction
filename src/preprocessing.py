
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(
    "dataset/archive/location_aware_gis_leakage_dataset_with_units.csv"
)

# Select input features
features = [
    "Pressure_PSI",
    "Flow_Rate_Lmin",
    "Temperature_C",
    "Vibration_mms",
    "RPM",
    "Operational_Hours_h"
]

X = df[features]
y = df["Leakage_Flag"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save processed datasets
train_data = X_train.copy()
train_data["Leakage_Flag"] = y_train

test_data = X_test.copy()
test_data["Leakage_Flag"] = y_test

train_data.to_csv("outputs/train_data.csv", index=False)
test_data.to_csv("outputs/test_data.csv", index=False)

print("DATA PREPROCESSING COMPLETED!")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining leakage distribution:")
print(y_train.value_counts())

print("\nTesting leakage distribution:")
print(y_test.value_counts())
